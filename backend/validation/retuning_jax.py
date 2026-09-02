"""JAX/GPU port of the Section 4.2 step-response evaluation.

The NumPy path evaluates one gain pair at a time in a Python loop, which is what
makes a 2,805-candidate hierarchical grid search cost ~10 minutes per cell. The
whole search is embarrassingly parallel across candidates and identical in
control flow, so it maps directly onto ``vmap`` + ``jit``: every candidate
advances through the same 6,000 RK4 steps in lockstep, as one batched kernel.

This module is a **reimplementation, not a wrapper**. It must therefore be held
against the original: ``tests/test_retuning_jax.py`` asserts agreement with
``backend.validation.retuning.evaluate_gains`` on every campaign plant, and
``verify_against_numpy`` does the same at runtime. A fast simulator that
disagrees with the reference is worse than no simulator at all.

Everything here mirrors the NumPy configuration the campaign actually uses:
the paper velocity gain enabled, the EA gain cap and the correction clamp off,
feedforward on from the velocity reference, and the control period equal to the
integration step.

Setup note: ``jax[cuda12]`` needs the pip-installed CUDA libraries ahead of the
system ``/usr/local/cuda/lib64`` on ``LD_LIBRARY_PATH``, or it silently falls
back to CPU. ``configure_gpu_env()`` builds the right value.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
def nvidia_lib_dirs() -> list[str]:
    """The pip-installed CUDA library directories inside this interpreter."""

    root = Path(sys.prefix)
    out: list[str] = []
    for base in (root / "lib64", root / "lib"):
        nvidia = base / f"python3.{sys.version_info.minor}" / "site-packages" / "nvidia"
        if nvidia.is_dir():
            out.extend(str(p) for p in sorted(nvidia.glob("*/lib")) if p.is_dir())
    return out


def configure_gpu_env() -> str:
    """Return an ``LD_LIBRARY_PATH`` that lets JAX find the GPU.

    The system CUDA 13.3 tree on this machine shadows the CUDA 12 libraries the
    jax wheels ship, and the failure is silent: JAX reports "Could not find cuda
    drivers" and quietly runs on CPU. Putting the wheel directories first, with
    the driver's own /usr/lib64 last, fixes it.
    """

    parts = [*nvidia_lib_dirs(), "/usr/lib64"]
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if current:
        parts.extend(p for p in current.split(":") if p and "cuda" not in p)
    return ":".join(parts)


# --------------------------------------------------------------------------- #
# parameter packing
# --------------------------------------------------------------------------- #
PARAM_FIELDS = (
    "EA", "L1", "L2", "L3", "R1", "R2", "R3",
    "J1", "J2", "J3", "f1", "f2", "f3", "v_feed",
)


def pack_plant(params, line_speed_m_s: float) -> np.ndarray:
    """Flatten an ``R2RParameters`` into the vector the kernels take."""

    return np.array([
        params.EA,
        *params.span_length_m,
        *params.roller_radius_m,
        *params.inertia_kg_m2,
        *params.kf,
        float(line_speed_m_s),
    ], dtype=np.float64)


# --------------------------------------------------------------------------- #
# kernels
# --------------------------------------------------------------------------- #
def build_kernels(n_steps: int, dt: float, settle_index: int, last_edge_index: int,
                  overshoot_weight: float = 2.0):
    """Return ``(batch_cost, single_cost)`` closed over the episode shape.

    ``n_steps`` and the two window indices are static so the scan length and the
    scoring windows are compile-time constants; changing the record recompiles.
    The reference trajectory is passed in as an array, so the same kernel scores
    any schedule without a recompile.
    """

    import jax
    import jax.numpy as jnp

    def _unpack(p):
        return (p[0], p[1:4], p[4:7], p[7:10], p[10:13], p[13])

    def derivatives(x, u, p):
        EA, L, R, J, f, v_feed = _unpack(p)
        T = x[:3]
        w = x[3:]
        v = R * w                       # roller surface speeds
        # Eq. (1): span tension transport. Spans are UW->feeder, feeder->Nip,
        # Nip->RW, with the velocity-master feeder between spans 1 and 2.
        dT1 = (EA / L[0]) * (v_feed - v[0]) - (v_feed / L[0]) * T[0]
        dT2 = (EA / L[1]) * (v[1] - v_feed) + (T[0] * v_feed - T[1] * v[1]) / L[1]
        dT3 = (EA / L[2]) * (v[2] - v[1]) + (T[1] * v[1] - T[2] * v[2]) / L[2]
        # Eq. (2): roller torque balance. Web torque is R_i * (T_{i+1} - T_i)
        # with the topology's tension differences (-T1, T2-T3, T3).
        dw1 = (u[0] - T[0] * R[0] - f[0] * w[0]) / J[0]
        dw2 = (u[1] + T[1] * R[1] - T[2] * R[1] - f[1] * w[1]) / J[1]
        dw3 = (u[2] + T[2] * R[2] - f[2] * w[2]) / J[2]
        return jnp.stack([dT1, dT2, dT3, dw1, dw2, dw3])

    def rk4(x, u, p):
        k1 = derivatives(x, u, p)
        k2 = derivatives(x + 0.5 * dt * k1, u, p)
        k3 = derivatives(x + 0.5 * dt * k2, u, p)
        k4 = derivatives(x + dt * k3, u, p)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def steady_v(p, target):
        EA, L, R, J, f, v_feed = _unpack(p)
        v_uw = v_feed * (1.0 - target[0] / EA)
        v_nip = v_feed * (EA - target[0]) / jnp.maximum(EA - target[1], 1e-9)
        v_rw = v_nip * (EA - target[1]) / jnp.maximum(EA - target[2], 1e-9)
        return jnp.stack([v_uw, v_nip, v_rw])

    POLARITY = jnp.array([-1.0, 1.0, 1.0])
    INTEGRAL_LIMIT = 200.0

    def control(x, integ, p, target, kp_star, ti):
        EA, L, R, J, f, v_feed = _unpack(p)
        T = x[:3]
        w = x[3:]
        err = target - T
        integ = jnp.clip(integ + POLARITY * err * dt, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)
        pi_out = POLARITY * err + integ / ti
        # No EA gain cap and no correction clamp: both are dashboard safety
        # heuristics that truncate a gain search (see retuning.py).
        correction = (L / EA) * kp_star * pi_out
        v_ref = steady_v(p, target) + correction
        w_ref = v_ref / R
        w_err = w_ref - w
        tau_web = R * jnp.stack([-T[0], T[1] - T[2], T[2]])
        ff = f * w_ref - tau_web                     # feedforward from the reference
        k_vel = 1.4 * J * jnp.sqrt(EA * R * R / (J * L))   # the paper's K_vel
        return k_vel * w_err + ff, integ

    def nominal(p, target):
        EA, L, R, J, f, v_feed = _unpack(p)
        return jnp.concatenate([target, steady_v(p, target) / R])

    def simulate(p, ref, kp_star, ti):
        """Run the full record against a time-varying reference."""

        x0 = nominal(p, ref[0])
        integ0 = jnp.zeros(3)

        def body(carry, k):
            x, integ = carry
            u, integ = control(x, integ, p, ref[k], kp_star, ti)
            x = rk4(x, u, p)
            return (x, integ), x[:3]

        _, trace = jax.lax.scan(body, (x0, integ0), jnp.arange(n_steps))
        # scan records the state *after* each step; prepend the initial sample
        # so index k is the tension at time k*dt, matching the NumPy logger.
        return jnp.concatenate([x0[:3][None, :], trace], axis=0)

    SCALE_E, SCALE_OS, SCALE_T = 3.0, 20.0, 3.0
    W_OS = float(overshoot_weight)
    BAND = 0.02

    def cost(p, ref, step_size, kp_star, ti):
        trace = simulate(p, ref, kp_star, ti)
        idx = jnp.arange(trace.shape[0])
        post = idx >= settle_index
        n_post = jnp.sum(post)

        # --- tracking: all three channels against the time-varying reference -
        err = trace - ref
        sq = jnp.where(post[:, None], err ** 2, 0.0)
        rmse_y = jnp.mean(jnp.sqrt(jnp.sum(sq, axis=0) / n_post))

        # --- overshoot: excursion past the commanded level, in the direction
        #     that level was last moved. The direction is carried forward from
        #     each edge with a running max over the sign of dref.
        dref = jnp.diff(ref, axis=0, prepend=ref[:1])
        sign = jnp.sign(dref)
        # forward-fill the last non-zero sign per channel
        def ffill(carry, row):
            carry = jnp.where(row != 0.0, row, carry)
            return carry, carry
        _, direction = jax.lax.scan(ffill, jnp.zeros(3), sign)
        excess = err * direction
        os_pct = jnp.max(jnp.where(post[:, None] & (direction != 0.0),
                                   jnp.maximum(excess, 0.0), 0.0)) * 100.0 / step_size

        # --- settling: measured from the final edge -------------------------
        band = 0.02 * step_size
        tail = idx >= last_edge_index
        outside = (jnp.abs(err) > band) & tail[:, None]
        any_out = jnp.any(outside, axis=1)
        last_exit = jnp.max(jnp.where(any_out, idx, -1))
        horizon = (trace.shape[0] - 1 - last_edge_index) * dt
        never = last_exit >= (trace.shape[0] - 1)
        settled_t = jnp.where(last_exit < 0, 0.0,
                              jnp.where(never, horizon,
                                        (last_exit + 1 - last_edge_index) * dt))

        S = ((rmse_y / SCALE_E) ** 2
             + W_OS * (os_pct / SCALE_OS) ** 2
             + (settled_t / SCALE_T) ** 2)
        return jnp.stack([S, rmse_y, os_pct, settled_t])

    single = jax.jit(cost)
    # Batch over gain pairs only; the plant and targets are shared per cell.
    batch = jax.jit(jax.vmap(cost, in_axes=(None, None, None, 0, 0)))
    return batch, single


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
class JaxEvaluator:
    """Batched record cost for one plant, on whatever backend JAX has."""

    def __init__(self, params, line_speed_m_s: float, *, reference: np.ndarray,
                 dt: float, settle_index: int, last_edge_index: int,
                 step_size_N: float):
        import jax.numpy as jnp

        self.dt = dt
        n_steps = reference.shape[0] - 1
        from .retuning import overshoot_weight
        self.batch, self.single = build_kernels(n_steps, dt, settle_index,
                                               last_edge_index, overshoot_weight())
        self.p = jnp.asarray(pack_plant(params, line_speed_m_s))
        self.ref = jnp.asarray(np.asarray(reference, dtype=np.float64))
        self.step_size = float(step_size_N)

    @classmethod
    def for_campaign(cls, params, line_speed_m_s: float) -> "JaxEvaluator":
        """Build the evaluator straight from the campaign's own schedule."""

        from .paper_inputs import excitation_schedule
        from .excitations import get_excitation_profile
        from .retuning import (RETUNING_CAMPAIGN_GROUP, RETUNING_EXCITATION,
                               STEP_FRACTION)

        schedule = excitation_schedule(RETUNING_EXCITATION, RETUNING_CAMPAIGN_GROUP, 0)
        base = np.asarray(params.tension_ref_N, dtype=np.float64)
        step_n = STEP_FRACTION * float(base[0])
        profile = get_excitation_profile(RETUNING_EXCITATION, step_n,
                                         campaign_group=RETUNING_CAMPAIGN_GROUP)
        dt = float(schedule.dt_s)
        n = int(round(float(schedule.duration_s) / dt))
        ref = np.array([base + np.asarray(profile(k * dt)) for k in range(n + 1)])
        return cls(params, line_speed_m_s, reference=ref, dt=dt,
                   settle_index=int(round(float(schedule.settle_s) / dt)),
                   last_edge_index=int(round(max(e.time_s for e in schedule.edges) / dt)),
                   step_size_N=step_n)

    def evaluate(self, kp: np.ndarray, ti: np.ndarray) -> np.ndarray:
        """Score a batch of gain pairs. Returns (n, 4): S, RMSE, OS%, t_s."""

        import jax.numpy as jnp

        out = self.batch(self.p, self.ref, self.step_size,
                         jnp.asarray(np.asarray(kp, dtype=np.float64)),
                         jnp.asarray(np.asarray(ti, dtype=np.float64)))
        return np.asarray(out)

    def cost_only(self, kp: np.ndarray, ti: np.ndarray) -> np.ndarray:
        return self.evaluate(kp, ti)[:, 0]
