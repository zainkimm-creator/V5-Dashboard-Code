"""Candidate evaluation models for the Section 4.2 retuning cost.

The papers do not pin down how the cost S of Eq. (12) is measured — which test
signal, whether the field sensor noise is in the loop during the real-plant
evaluations, and whether the metrics are formed from true or measured tension.
Those choices span a ~50x range in S, so instead of guessing, this module makes
each reading an explicit, runnable model, and ``anchor_sweep`` scores every model
against pre-registered anchors taken from the paper's own numbers. Models that
fail the anchors are falsified, not tuned.

Axes (each combination is one ``EvalModel``):

* ``signal``     - what the plant is asked to do while S is measured:
                   ``record``  (the campaign's 16 s staggered E_Toggle schedule),
                   ``step1``   (a single +20 % step on channel 1, 1 s settle + 5 s),
                   ``step3``   (a simultaneous +20 % step on all three channels),
                   ``stepseq`` (each channel stepped +20 % in turn and held -
                   edges at 1/6/11 s, one channel per episode, no compound
                   edges; gives every channel its own |dT_ref,i| as Eq. (12)'s
                   per-channel normalisers require).
* ``noise``      - field-matched sensor noise in the control loop during the
                   evaluation (the plant "as it runs in the field"), or a clean
                   loop. With noise on, the controller sees LPF-filtered noisy
                   measurements exactly as ``simulation.py`` builds them.
* ``cost_source``- metrics formed from ``true`` tensions or from the ``measured``
                   (noisy, filtered) tensions sampled at T_log - on a real line
                   only the latter exists.
* ``integral_clamp`` - the dashboard clamps the outer-loop integral at
                   +-200 N s for every plant; the paper prints no clamp.

Everything runs double-precision on the JAX backend; the noise-free ``record`` /
``true`` / clamped model must agree with the NumPy reference
(``retuning.evaluate_gains``) — asserted in tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Sequence

import numpy as np

from .retuning_jax import pack_plant

# Field-matched acquisition (Table S9 header): dual-channel 0.3 % noise, 50/50 Hz.
FIELD_TENSION_NOISE_FRACTION = 0.003   # of T_max = T_ref / 0.30
FIELD_VELOCITY_NOISE_FRACTION = 0.003  # of v_max = v0 / 0.30, as surface speed
FIELD_LPF_HZ = 50.0
FIELD_TLOG_S = 0.005


@dataclass(frozen=True)
class EvalModel:
    """One candidate reading of how the retuning cost is evaluated."""

    signal: str = "record"          # record | step1 | step3
    noise: bool = False             # field-matched noise in the loop
    lpf_in_loop: bool = False       # 50 Hz LPF on the feedback WITHOUT noise:
                                    # "T_meas ... is the low-pass-filtered signal
                                    # actually available to the drive" (main p.7,
                                    # Sec. 2.3) - deterministic, so consistent
                                    # with the documented determinism of the HGS
                                    # arms (retuning_reference.json seed_structure)
    cost_source: str = "true"       # true | measured
    integral_clamp: float | None = 200.0   # N*s; None = no clamp (paper prints none)
    w_os: float = 2.0               # Eq. (12) overshoot weight (published: 2)
    noise_seed: int = 0

    @property
    def key(self) -> str:
        parts = [self.signal,
                 "noise" if self.noise else ("lpf" if self.lpf_in_loop else "clean"),
                 self.cost_source,
                 "clamp" if self.integral_clamp is not None else "noclamp"]
        if self.w_os != 2.0:
            parts.append(f"wos{self.w_os:g}")
        return "-".join(parts)

    def __post_init__(self) -> None:
        if self.signal not in ("record", "step1", "step3", "stepseq"):
            raise ValueError(f"unknown signal {self.signal!r}")
        if self.cost_source not in ("true", "measured"):
            raise ValueError(f"unknown cost_source {self.cost_source!r}")
        if self.cost_source == "measured" and not self.noise:
            # Measured == what the field sensors deliver; without the noise/LPF
            # chain there is no distinct measured signal.
            raise ValueError("cost_source='measured' requires noise=True")


# --------------------------------------------------------------------------- #
# reference construction
# --------------------------------------------------------------------------- #
def build_reference(params, model: EvalModel) -> tuple[np.ndarray, float, int, int, float]:
    """Return (reference trace, dt, settle_index, last_edge_index, step_size_N).

    The reference is the tension setpoint each channel is commanded to hold,
    sampled on the integration grid, one row per step plus the initial sample.
    """

    base = np.asarray(params.tension_ref_N, dtype=np.float64)
    step_n = 0.20 * float(base[0])

    if model.signal == "record":
        from .excitations import get_excitation_profile
        from .paper_inputs import excitation_schedule

        sched = excitation_schedule("E_Toggle", "C_retuning_field_matched", 0)
        dt = float(sched.dt_s)
        n = int(round(float(sched.duration_s) / dt))
        profile = get_excitation_profile("E_Toggle", step_n,
                                         campaign_group="C_retuning_field_matched")
        ref = np.array([base + np.asarray(profile(k * dt)) for k in range(n + 1)])
        settle_i = int(round(float(sched.settle_s) / dt))
        last_i = int(round(max(e.time_s for e in sched.edges) / dt))
        return ref, dt, settle_i, last_i, step_n

    dt = 0.001
    settle_s, episode_s = 1.0, 5.0
    if model.signal == "stepseq":
        # One channel per 5 s episode, held (cumulative): 1 s settle, edges at
        # 1 / 6 / 11 s, 16 s total. Settling is measured from the last edge.
        n = int(round((settle_s + 3 * episode_s) / dt))
        edges = [settle_s, settle_s + episode_s, settle_s + 2 * episode_s]
        ref = np.array([base + step_n * np.array([k * dt >= e for e in edges],
                                                 dtype=np.float64)
                        for k in range(n + 1)])
        return (ref, dt, int(round(settle_s / dt)),
                int(round(edges[2] / dt)), step_n)

    # Single simultaneous step tests: 1 s settle + 5 s episode (S1.1 timing).
    n = int(round((settle_s + episode_s) / dt))
    mask = np.array([1.0, 0.0, 0.0]) if model.signal == "step1" else np.ones(3)
    deltas = step_n * mask
    ref = np.array([base + (deltas if k * dt >= settle_s else 0.0)
                    for k in range(n + 1)])
    settle_i = int(round(settle_s / dt))
    return ref, dt, settle_i, settle_i, step_n


# --------------------------------------------------------------------------- #
# kernel
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=64)
def build_model_kernel(model: EvalModel, n_steps: int, dt: float,
                       settle_index: int, last_edge_index: int):
    """JIT-compiled ``cost(p, ref, step_size, noiseT, noiseW, kp, ti) -> (4,)``.

    The closed loop mirrors ``simulation.py``: measured tension = LPF(true +
    noise) with alpha = 1 - exp(-2*pi*fc*dt); per-roller omega noise is the
    surface-speed sigma divided by that roller's radius, filtered the same way.
    With ``model.noise`` off the controller sees the true state directly - no
    filter in the loop - which is the configuration the NumPy reference runs.
    """

    import jax
    import jax.numpy as jnp

    # The filter path runs whenever the drive sees filtered feedback - with
    # noise (field evaluation) or without (deterministic LPF-lagged loop).
    NOISE = bool(model.noise or model.lpf_in_loop)
    MEASURED_COST = model.cost_source == "measured"
    CLAMP = model.integral_clamp
    W_OS = float(model.w_os)
    TLOG_STRIDE = max(1, int(round(FIELD_TLOG_S / dt))) if MEASURED_COST else 1
    ALPHA = 1.0 - math.exp(-2.0 * math.pi * FIELD_LPF_HZ * dt)

    def _unpack(p):
        return (p[0], p[1:4], p[4:7], p[7:10], p[10:13], p[13])

    def derivatives(x, u, p):
        EA, L, R, J, f, v_feed = _unpack(p)
        T, w = x[:3], x[3:]
        v = R * w
        dT1 = (EA / L[0]) * (v_feed - v[0]) - (v_feed / L[0]) * T[0]
        dT2 = (EA / L[1]) * (v[1] - v_feed) + (T[0] * v_feed - T[1] * v[1]) / L[1]
        dT3 = (EA / L[2]) * (v[2] - v[1]) + (T[1] * v[1] - T[2] * v[2]) / L[2]
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

    def control(T_meas, w_meas, integ, p, target, kp_star, ti):
        EA, L, R, J, f, v_feed = _unpack(p)
        err = target - T_meas
        integ = integ + POLARITY * err * dt
        if CLAMP is not None:
            integ = jnp.clip(integ, -CLAMP, CLAMP)
        pi_out = POLARITY * err + integ / ti
        correction = (L / EA) * kp_star * pi_out
        v_ref = steady_v(p, target) + correction
        w_ref = v_ref / R
        tau_web = R * jnp.stack([-T_meas[0], T_meas[1] - T_meas[2], T_meas[2]])
        ff = f * w_ref - tau_web
        k_vel = 1.4 * J * jnp.sqrt(EA * R * R / (J * L))
        return k_vel * (w_ref - w_meas) + ff, integ

    def cost(p, ref, step_size, noiseT, noiseW, kp_star, ti):
        EA, L, R, J, f, v_feed = _unpack(p)
        x0 = jnp.concatenate([ref[0], steady_v(p, ref[0]) / R])

        def body(carry, k):
            x, integ, fT, fW = carry
            if NOISE:
                rawT = x[:3] + noiseT[k]
                fT = fT + ALPHA * (rawT - fT)
                rawW = x[3:] + noiseW[k]
                fW = fW + ALPHA * (rawW - fW)
                T_meas, w_meas = fT, fW
            else:
                T_meas, w_meas = x[:3], x[3:]
            u, integ = control(T_meas, w_meas, integ, p, ref[k], kp_star, ti)
            x = rk4(x, u, p)
            return (x, integ, fT, fW), (x[:3], T_meas)

        carry0 = (x0, jnp.zeros(3), x0[:3], x0[3:])
        _, (trace_true, trace_meas) = jax.lax.scan(body, carry0, jnp.arange(n_steps))
        trace_true = jnp.concatenate([x0[:3][None, :], trace_true], axis=0)
        trace_meas = jnp.concatenate([x0[:3][None, :], trace_meas], axis=0)
        trace = trace_meas if MEASURED_COST else trace_true

        idx = jnp.arange(trace.shape[0])
        # Metrics on the T_log grid when scoring the measured signal - on a real
        # line only logged samples exist.
        on_grid = (idx % TLOG_STRIDE) == 0
        post = (idx >= settle_index) & on_grid
        n_post = jnp.sum(post)
        err = trace - ref

        sq = jnp.where(post[:, None], err ** 2, 0.0)
        rmse_y = jnp.mean(jnp.sqrt(jnp.sum(sq, axis=0) / n_post))

        dref = jnp.diff(ref, axis=0, prepend=ref[:1])
        sign = jnp.sign(dref)

        def ffill(carry, row):
            carry = jnp.where(row != 0.0, row, carry)
            return carry, carry

        _, direction = jax.lax.scan(ffill, jnp.zeros(3), sign)
        excess = err * direction
        os_pct = jnp.max(jnp.where(post[:, None] & (direction != 0.0),
                                   jnp.maximum(excess, 0.0), 0.0)) * 100.0 / step_size

        band = 0.02 * step_size
        tail = (idx >= last_edge_index) & on_grid
        outside = (jnp.abs(err) > band) & tail[:, None]
        any_out = jnp.any(outside, axis=1)
        last_exit = jnp.max(jnp.where(any_out, idx, -1))
        horizon = (trace.shape[0] - 1 - last_edge_index) * dt
        never = last_exit >= (trace.shape[0] - 1 - (TLOG_STRIDE - 1))
        settled_t = jnp.where(last_exit < 0, 0.0,
                              jnp.where(never, horizon,
                                        (last_exit + 1 - last_edge_index) * dt))

        S = ((rmse_y / 3.0) ** 2 + W_OS * (os_pct / 20.0) ** 2
             + (settled_t / 3.0) ** 2)
        return jnp.stack([S, rmse_y, os_pct, settled_t])

    batch = jax.jit(jax.vmap(cost, in_axes=(None, None, None, None, None, 0, 0)))
    single = jax.jit(cost)
    return batch, single


# --------------------------------------------------------------------------- #
# evaluator
# --------------------------------------------------------------------------- #
class ModelEvaluator:
    """One (plant, model) pair, ready to score gain batches on the GPU."""

    def __init__(self, params, line_speed_m_s: float, model: EvalModel):
        import jax
        import jax.numpy as jnp

        self.model = model
        ref, dt, settle_i, last_i, step_n = build_reference(params, model)
        self.dt, self.step_n = dt, step_n
        self.batch, self.single = build_model_kernel(model, ref.shape[0] - 1, dt,
                                                     settle_i, last_i)
        self.p = jnp.asarray(pack_plant(params, line_speed_m_s))
        self.ref = jnp.asarray(ref)
        n = ref.shape[0] - 1

        if model.noise:
            t_max = float(params.tension_ref_N[0]) / 0.30
            v_max = float(line_speed_m_s) / 0.30
            sigma_t = FIELD_TENSION_NOISE_FRACTION * t_max
            sigma_v = FIELD_VELOCITY_NOISE_FRACTION * v_max
            radii = np.asarray(params.roller_radius_m, dtype=np.float64)
            key = jax.random.PRNGKey(model.noise_seed)
            kT, kW = jax.random.split(key)
            self.noiseT = sigma_t * jax.random.normal(kT, (n, 3), dtype=jnp.float64)
            self.noiseW = (sigma_v / radii)[None, :] * jax.random.normal(
                kW, (n, 3), dtype=jnp.float64)
        else:
            self.noiseT = jnp.zeros((n, 3))
            self.noiseW = jnp.zeros((n, 3))

    def with_noise_seed(self, seed: int) -> "ModelEvaluator":
        """Fresh noise realisation (a new physical run on the same plant)."""

        import copy
        import jax
        import jax.numpy as jnp

        if not self.model.noise:
            return self
        out = copy.copy(self)
        n = self.ref.shape[0] - 1
        sigT = float(jnp.std(self.noiseT[:, 0])) if n else 0.0
        key = jax.random.PRNGKey(seed)
        kT, kW = jax.random.split(key)
        scaleW = jnp.std(self.noiseW, axis=0)
        out.noiseT = sigT * jax.random.normal(kT, (n, 3), dtype=jnp.float64)
        out.noiseW = scaleW[None, :] * jax.random.normal(kW, (n, 3), dtype=jnp.float64)
        return out

    def evaluate(self, kp: np.ndarray, ti: np.ndarray) -> np.ndarray:
        import jax.numpy as jnp

        out = self.batch(self.p, self.ref, self.step_n, self.noiseT, self.noiseW,
                         jnp.asarray(np.asarray(kp, dtype=np.float64)),
                         jnp.asarray(np.asarray(ti, dtype=np.float64)))
        return np.asarray(out)

    def cost_only(self, kp, ti) -> np.ndarray:
        return self.evaluate(kp, ti)[:, 0]
