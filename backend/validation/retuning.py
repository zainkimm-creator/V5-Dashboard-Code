"""Section 4.2 retuning: campaign definition, cost function and gain search.

This is the Tier 2 side of Section 4 - it *runs* the optimizer rather than
re-reading published numbers. Paper values stay comparison-only; nothing here
reads them.

The campaign
------------
Six plants x ten drift scenarios = 60 combinations, under two identification
protocols. For each combination a digital twin is identified from a single
excitation record, five retuning strategies search for a PI gain pair, and the
resulting gains are scored on the true drifted plant.

Every constant below is sourced, because most of them are the kind that
silently change a result:

* **Plants** - main-text Table 2: P001, P049, P053, P158, P186, P189 (the five
  slow-transport plants plus the fast plant P189 as a documented worst case).
* **Drift scenarios** - Section 3.3 and the Fig. 4 source data: nine
  single-family legs D01-D09 plus the combined D10. Friction moves at +-30%
  only; the +-15% legs that exist in ``data/processed`` are not campaign cells.
* **Cost** - Eq. (12) with fixed scales 3 N / 20 % / 3 s and overshoot weight 2.
* **Step response** - supplement S1.1: retuning campaigns settle for 1 s, step
  every channel to +20% of its setpoint, and run 5 s episodes.
* **HGS budget** - supplement S8.1: ~2,805 twin evaluations as a coarse grid, a
  Latin-hypercube refinement, a fine grid and a five-evaluation BO refinement.
* **Protocols** - Table S9 / ledger campaigns 10 and 11.

Transfer rule
-------------
HGS hands the real-plant optimizer candidate gain **locations only**. The twin's
cost values are withheld, because seeding a real-plant surrogate with simulated
costs would bias it by exactly the sim-to-real gap the experiment is measuring
(main text 4.2). ``hgs_candidates`` therefore returns points, not scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Mapping, Sequence

from ..models.controller import ControllerConfig, auto_tension_integral_time_s
from ..models.equations import R2RParameters
from ..models.simulation import SimulationConfig, simulate
from .plants import parameters_for_plant

# --------------------------------------------------------------------------- #
# campaign definition
# --------------------------------------------------------------------------- #
# (pool id, dashboard id). Table 2's six-plant subset.
RETUNING_PLANTS: tuple[tuple[str, str], ...] = (
    ("P001", "P01"),
    ("P049", "P02"),
    ("P053", "P03"),
    ("P158", "P06"),
    ("P186", "P09"),
    ("P189", "P10"),
)
# The five slow-transport plants the paper anchors its win rate on; P189 is the
# fast-plant exception reported separately.
ANCHORED_5 = ("P001", "P049", "P053", "P158", "P186")
FAST_EXCEPTION = "P189"


@dataclass(frozen=True)
class DriftScenario:
    """One drift cell. Scales are absolute multipliers on the pre-drift plant."""

    code: str
    label: str
    EA_scale: float = 1.0
    friction_scale: float = 1.0
    J_UW_scale: float = 1.0
    J_Nip_scale: float = 1.0
    J_RW_scale: float = 1.0


# Nine single-family legs plus the combined tenth (Section 3.3, Fig. 4).
DRIFT_SCENARIOS: tuple[DriftScenario, ...] = (
    DriftScenario("D01", "EA +10%", EA_scale=1.10),
    DriftScenario("D02", "EA -10%", EA_scale=0.90),
    DriftScenario("D03", "EA +30%", EA_scale=1.30),
    DriftScenario("D04", "EA -30%", EA_scale=0.70),
    DriftScenario("D05", "EA +50%", EA_scale=1.50),
    DriftScenario("D06", "J: UW -30% / RW +50%", J_UW_scale=0.70, J_RW_scale=1.50),
    DriftScenario("D07", "J: UW -50% / RW +100%", J_UW_scale=0.50, J_RW_scale=2.00),
    DriftScenario("D08", "f +30%", friction_scale=1.30),
    DriftScenario("D09", "f -30%", friction_scale=0.70),
    DriftScenario("D10", "EA +20%, J_UW -20%, f +15%",
                  EA_scale=1.20, friction_scale=1.15, J_UW_scale=0.80),
)
DRIFT_BY_CODE = {s.code: s for s in DRIFT_SCENARIOS}


@dataclass(frozen=True)
class Protocol:
    """A SysID acquisition protocol. Only the identification differs between the
    two campaigns; the retuning grid is identical (ledger campaigns 10 and 11)."""

    name: str
    log_sample_time_s: float
    tension_noise_fraction: float  # of T_max
    velocity_noise_fraction: float  # of v_max; 0 => tension-only
    tension_lpf_hz: float
    velocity_lpf_hz: float | None
    record_duration_s: float = 16.0


# Table S9 header and the §4.2 text: the field-matched protocol matches the noise
# the plant actually sees; logging-only sits inside the noise-aware window but
# carries no velocity-channel noise.
PROTOCOL_FIELD_MATCHED = Protocol(
    "field_matched", log_sample_time_s=0.005,
    tension_noise_fraction=0.003, velocity_noise_fraction=0.003,
    tension_lpf_hz=50.0, velocity_lpf_hz=50.0,
)
PROTOCOL_LOGGING_ONLY = Protocol(
    "logging_only", log_sample_time_s=0.020,
    tension_noise_fraction=0.003, velocity_noise_fraction=0.0,
    tension_lpf_hz=100.0, velocity_lpf_hz=None,
)
PROTOCOLS = {p.name: p for p in (PROTOCOL_FIELD_MATCHED, PROTOCOL_LOGGING_ONLY)}

# Eq. (12) fixed scale constants and the overshoot doubling weight.
COST_SCALE_RMSE_N = 3.0
COST_SCALE_OVERSHOOT_PCT = 20.0
COST_SCALE_SETTLING_S = 3.0
COST_OVERSHOOT_WEIGHT = 2.0
# Overriding the published overshoot weight is an EXPERIMENT, not a reading of
# the paper. Eq. (12) prints w_os = 2 and nothing here argues otherwise.
#
# It exists because the overshoot term is the single place this reproduction
# and the paper part company. On the campaign record no gain pair in a
# 5,400-point sweep gets overshoot under ~8 % without cost exploding (S = 3-65),
# yet the published median of 0.357 requires OS <= 8.4 % from that term alone.
# Setting the weight to 0 asks "what would the numbers be if our overshoot
# matched theirs?" - and the answer lands on the paper (see the report). Any
# result produced with this set must be labelled as such.
COST_OVERSHOOT_WEIGHT_OVERRIDE: float | None = None


def overshoot_weight() -> float:
    return (COST_OVERSHOOT_WEIGHT if COST_OVERSHOOT_WEIGHT_OVERRIDE is None
            else float(COST_OVERSHOOT_WEIGHT_OVERRIDE))

# The retuning campaigns score a full E_Toggle record, not a single step.
#
# `excitation_schedules_v5.csv` prints the schedule for campaign group
# `C_retuning_field_matched` and marks it "used by ledger campaigns 10-11" -
# which are the two retuning campaigns. It is a 16 s staggered record with a 1 s
# settle and three step events:
#
#     t =  1 s   span1 UP
#     t =  6 s   span1 DOWN, span2 UP
#     t = 11 s   span1 UP,   span2 DOWN, span3 UP
#
# Scoring a single isolated step instead (an earlier reading here) leaves the
# record with one transient rather than six and puts the cost far below the
# published range, because Eq. (12) is dominated by transient tracking error.
RETUNING_EXCITATION = "E_Toggle"
RETUNING_CAMPAIGN_GROUP = "C_retuning_field_matched"
STEP_SETTLE_S = 1.0
STEP_EPISODE_S = 5.0
STEP_FRACTION = 0.20
# Which tension channels the step-response test moves.
#
# The paper does not state this for the §4.2 test, and it decides the result.
# Stepping all three together is physically inconsistent with the published
# numbers: tension cannot jump, so on a 270 N plant a simultaneous 54 N step
# forces RMSE_y ~ 4.9 N from the transient alone, and with s_e = 3 N *any*
# controller then scores >= 2.6 - against a reported pooled median of 0.357
# over plants up to 360 N. Stepping one channel, the structure E_Toggle
# actually uses (v5.1 corrected its own text from "all channels" to a
# staggered toggle, one channel per episode), puts every plant back in range.
# RMSE is still averaged over all three channels, so the untouched two
# contribute their coupling error, exactly as Eq. (12) specifies.
STEP_CHANNELS: tuple[bool, bool, bool] = (True, False, False)
SETTLING_BAND_FRACTION = 0.02  # Eq. (12) uses the 2 %-settling time, not t90.

# SysID-mode operating point (Section 4.1). This is also the WS-BO warm start:
# chosen for identifiability, not for closed-loop cost.
SYSID_MODE_KP = 100.0

# Gain search space. The retuned optimum sits near K_p* ~ 15 (§4.2), so the box
# must span roughly two decades below the SysID-mode gain to contain it.
KP_BOUNDS = (0.5, 300.0)
# Multiples of the plant's own auto_Ti. Widened after a pilot cell pinned the
# incumbent at a scale of 10: a bound the optimum sits on is a truncated search,
# not a converged one.
TI_SCALE_BOUNDS = (0.02, 100.0)

# Supplement S8.1 budget.
HGS_COARSE = 900
HGS_LHS = 1000
HGS_FINE = 900
HGS_DT_BO = 5
HGS_TWIN_BUDGET = HGS_COARSE + HGS_LHS + HGS_FINE + HGS_DT_BO  # 2805


# --------------------------------------------------------------------------- #
# drift
# --------------------------------------------------------------------------- #
def apply_drift(params: R2RParameters, scenario: DriftScenario) -> R2RParameters:
    """Return the drifted plant. Drift is a step change, not a ramp: the
    controller keeps its pre-drift gains and the plant has already moved."""

    moved = params.with_drift(
        EA_scale=scenario.EA_scale, friction_scale=scenario.friction_scale
    )
    return replace(
        moved,
        inertia_kg_m2=(
            moved.inertia_kg_m2[0] * scenario.J_UW_scale,
            moved.inertia_kg_m2[1] * scenario.J_Nip_scale,
            moved.inertia_kg_m2[2] * scenario.J_RW_scale,
        ),
    )


# --------------------------------------------------------------------------- #
# the cost function - Eq. (12)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CostBreakdown:
    """The scalar cost plus the three metrics that formed it."""

    S: float
    rmse_y_N: float
    overshoot_percent: float
    settling_time_s: float
    settled: bool

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "S": self.S,
            "RMSE_y_N": self.rmse_y_N,
            "OS_percent": self.overshoot_percent,
            "t_s_s": self.settling_time_s,
            "settled": self.settled,
        }


def retuning_reference_trace(
    params: R2RParameters, times_s: Sequence[float]
) -> list[tuple[float, float, float]]:
    """The setpoint each channel is asked to hold, sampled on ``times_s``.

    The campaign's own E_Toggle schedule added to the plant setpoints, so the
    cost is scored against exactly the reference the plant was commanded.
    """

    from .excitations import get_excitation_profile

    base = tuple(float(v) for v in params.tension_ref_N)
    profile = get_excitation_profile(
        RETUNING_EXCITATION, STEP_FRACTION * base[0],
        campaign_group=RETUNING_CAMPAIGN_GROUP,
    )
    out = []
    for t in times_s:
        d = profile(float(t))
        out.append((base[0] + d[0], base[1] + d[1], base[2] + d[2]))
    return out


def record_cost(
    times_s: Sequence[float],
    tensions_N: Sequence[Sequence[float]],
    reference_N: Sequence[Sequence[float]],
    *,
    settle_s: float,
    last_edge_s: float,
    step_size_N: float,
) -> CostBreakdown:
    """Score a full staggered record with Eq. (12).

    Differences from a single-step reading, all forced by the record's shape:

    * ``RMSE_y`` accumulates over the whole post-settle record against the
      *time-varying* reference, so every one of the six edges contributes its
      transient.
    * ``OS`` is the largest excursion past the commanded level on any channel
      after any edge, normalised by the step size (a constant +-20 % of setpoint).
    * ``t_s`` is measured from the final edge, the only point after which every
      channel has a level it must hold to the end of the record.
    """

    if not times_s or len(times_s) != len(tensions_N):
        raise ValueError("times and tensions must be non-empty and equal length")
    if step_size_N <= 0:
        raise ValueError("step_size_N must be positive")

    post = [i for i, t in enumerate(times_s) if t >= settle_s]
    if not post:
        raise ValueError("no samples after the settle window")

    per_channel = []
    for ch in range(3):
        acc = sum((float(tensions_N[i][ch]) - float(reference_N[i][ch])) ** 2
                  for i in post)
        per_channel.append(math.sqrt(acc / len(post)))
    rmse_y = sum(per_channel) / 3.0

    # Overshoot: the largest excursion past a commanded level, in the direction
    # that level was last moved. The direction has to be carried forward from
    # each edge - testing it against the previous sample only fires on the single
    # transition sample, which reports 0 % for every run.
    overshoot = 0.0
    for ch in range(3):
        direction = 0.0
        for i in post:
            ref = float(reference_N[i][ch])
            prev = float(reference_N[i - 1][ch]) if i > 0 else ref
            if ref != prev:
                direction = 1.0 if ref > prev else -1.0
            if direction == 0.0:
                continue
            excess = (float(tensions_N[i][ch]) - ref) * direction
            if excess > 0.0:
                overshoot = max(overshoot, 100.0 * excess / step_size_N)

    band = SETTLING_BAND_FRACTION * step_size_N
    tail = [i for i, t in enumerate(times_s) if t >= last_edge_s]
    last_exit = None
    for i in tail:
        if any(abs(float(tensions_N[i][ch]) - float(reference_N[i][ch])) > band
               for ch in range(3)):
            last_exit = i
    if last_exit is None:
        settling, settled = 0.0, True
    elif last_exit >= tail[-1]:
        settling, settled = float(times_s[-1]) - last_edge_s, False
    else:
        settling, settled = float(times_s[last_exit + 1]) - last_edge_s, True

    cost = ((rmse_y / COST_SCALE_RMSE_N) ** 2
            + overshoot_weight() * (overshoot / COST_SCALE_OVERSHOOT_PCT) ** 2
            + (settling / COST_SCALE_SETTLING_S) ** 2)
    return CostBreakdown(cost, rmse_y, overshoot, settling, settled)


def step_response_cost(
    times_s: Sequence[float],
    tensions_N: Sequence[Sequence[float]],
    target_before_N: Sequence[float],
    target_after_N: Sequence[float],
    step_time_s: float,
    *,
    horizon_s: float | None = None,
) -> CostBreakdown:
    """Score a three-channel step response with the paper's composite cost.

        S = (RMSE_y / 3)^2 + 2 (OS / 20)^2 + (t_s / 3)^2

    ``RMSE_y`` is the arithmetic mean of the three per-channel RMS tracking
    errors after the step, ``OS`` the largest per-channel percent overshoot
    relative to that channel's own step size, and ``t_s`` the 2 %-settling time
    at which *every* channel has entered and stayed inside its band.

    A run that never settles is charged the full horizon rather than being
    dropped, so a diverging gain pair scores badly instead of scoring nothing.
    """

    if len(times_s) != len(tensions_N):
        raise ValueError("times and tensions must have equal length")
    if not times_s:
        raise ValueError("empty step response")

    deltas = [float(a) - float(b) for a, b in zip(target_after_N, target_before_N)]
    stepped = [i for i, d in enumerate(deltas) if abs(d) > 0.0]
    if not stepped:
        raise ValueError("at least one channel must step to normalize by")

    post = [i for i, t in enumerate(times_s) if t >= step_time_s]
    if not post:
        raise ValueError("no samples after the step")
    end_s = float(times_s[-1]) if horizon_s is None else float(horizon_s)

    # --- tracking: per-channel RMS about the post-step setpoint, then mean ---
    per_channel_rms: list[float] = []
    for ch in range(3):
        acc = 0.0
        for i in post:
            err = float(tensions_N[i][ch]) - float(target_after_N[ch])
            acc += err * err
        per_channel_rms.append(math.sqrt(acc / len(post)))
    rmse_y = sum(per_channel_rms) / 3.0

    # --- overshoot: largest excursion beyond the new setpoint, per channel ---
    # Overshoot and settling are defined relative to a step, so only the
    # stepped channels can carry them; the others have no step size to
    # normalise by. Tracking error (above) still spans all three.
    overshoot = 0.0
    for ch in stepped:
        target = float(target_after_N[ch])
        delta = deltas[ch]
        # A downward step overshoots downward, so track the extremum in the
        # direction of travel rather than assuming a rising step.
        if delta > 0:
            peak = max(float(tensions_N[i][ch]) for i in post)
            excess = peak - target
        else:
            peak = min(float(tensions_N[i][ch]) for i in post)
            excess = target - peak
        overshoot = max(overshoot, 100.0 * max(0.0, excess) / abs(delta))

    # --- settling: last exit from the band, over all channels together -------
    bands = {ch: SETTLING_BAND_FRACTION * abs(deltas[ch]) for ch in stepped}
    last_exit_index: int | None = None
    for i in post:
        for ch in stepped:
            if abs(float(tensions_N[i][ch]) - float(target_after_N[ch])) > bands[ch]:
                last_exit_index = i
                break
    if last_exit_index is None:
        settling = 0.0
        settled = True
    elif last_exit_index >= post[-1]:
        # Still outside the band at the horizon: never settled.
        settling = end_s - step_time_s
        settled = False
    else:
        settling = float(times_s[last_exit_index + 1]) - step_time_s
        settled = True

    cost = (
        (rmse_y / COST_SCALE_RMSE_N) ** 2
        + COST_OVERSHOOT_WEIGHT * (overshoot / COST_SCALE_OVERSHOOT_PCT) ** 2
        + (settling / COST_SCALE_SETTLING_S) ** 2
    )
    return CostBreakdown(cost, rmse_y, overshoot, settling, settled)


# --------------------------------------------------------------------------- #
# step-response evaluation
# --------------------------------------------------------------------------- #
def _step_excitation(step_deltas_N: Sequence[float]) -> Callable[[float], tuple[float, float, float]]:
    """A single simultaneous step on all three channels at ``STEP_SETTLE_S``."""

    a, b, c = (float(v) for v in step_deltas_N)

    def profile(t_s: float) -> tuple[float, float, float]:
        return (a, b, c) if t_s >= STEP_SETTLE_S else (0.0, 0.0, 0.0)

    return profile


def evaluate_gains(
    params: R2RParameters,
    kp_star: float,
    ti_s: float,
    *,
    line_speed_m_s: float,
    dt_s: float = 0.001,
) -> CostBreakdown:
    """Run the campaign's 16 s staggered record and return its cost.

    The cost is formed from the plant's true tensions rather than a noisy
    measurement: it scores closed-loop behaviour, and the noise model belongs to
    the identification protocol, not to this metric.
    """

    from .excitations import get_excitation_profile
    from .paper_inputs import excitation_schedule

    if not math.isfinite(kp_star) or kp_star <= 0:
        raise ValueError(f"kp_star must be positive and finite, got {kp_star!r}")
    if not math.isfinite(ti_s) or ti_s <= 0:
        raise ValueError(f"ti_s must be positive and finite, got {ti_s!r}")

    schedule = excitation_schedule(RETUNING_EXCITATION, RETUNING_CAMPAIGN_GROUP, 0)
    base = tuple(float(v) for v in params.tension_ref_N)
    step_n = STEP_FRACTION * base[0]
    profile = get_excitation_profile(RETUNING_EXCITATION, step_n,
                                     campaign_group=RETUNING_CAMPAIGN_GROUP)

    config = SimulationConfig(
        duration_s=float(schedule.duration_s),
        dt_s=dt_s,
        controller_sample_time_s=dt_s,
        log_sample_time_s=dt_s,
        line_speed_m_s=line_speed_m_s,
        sensor_lpf_hz=None,
        controller_tracks_drift=False,
    )
    controller = ControllerConfig(
        target_tension_N=base,
        line_speed_m_s=line_speed_m_s,
        Kp_star_m_s_per_N=float(kp_star),
        TI_s=float(ti_s),
        paper_velocity_gain_enabled=True,
        high_ea_kp_cap_enabled=False,
        velocity_correction_limit_fraction=None,
        # steady_velocity_uses_measured_upstream stays at its default (False):
        # the measured-upstream variant was tested and reduced the coupling
        # disturbance by only 3.1 % while shifting the campaign median from
        # 2.919 to 2.922 - a failed hypothesis, reverted so every backend runs
        # the same plain controller.
    )
    result = simulate(params=params, controller_config=controller, config=config,
                      excitation=profile, write_output=False)

    times = [float(r["time_s"]) for r in result.rows]
    tensions = [(float(r["T1"]), float(r["T2"]), float(r["T3"])) for r in result.rows]
    if not all(all(map(math.isfinite, row)) for row in tensions):
        # A diverging plant is a legitimate search outcome; charge it heavily
        # rather than letting a NaN reach the surrogate.
        return CostBreakdown(float("inf"), float("inf"), float("inf"),
                             float(schedule.duration_s), False)

    reference = [(base[0] + d[0], base[1] + d[1], base[2] + d[2])
                 for d in (profile(t) for t in times)]
    last_edge = max(e.time_s for e in schedule.edges)
    return record_cost(times, tensions, reference,
                       settle_s=float(schedule.settle_s),
                       last_edge_s=float(last_edge),
                       step_size_N=step_n)


# Evaluation-model override for the campaign. None = the legacy record kernel;
# an EvalModel key (e.g. "step1-lpf-true-clamp") switches every cost evaluation
# in run_cell to that model. Identification is untouched - the SysID record and
# protocol are documented and never in question.
EVAL_MODEL_OVERRIDE: "str | None" = None


def _eval_model_from_key(key: str):
    from .retuning_eval_models import EvalModel

    signal, loop, source, clamp = key.split("-")
    return EvalModel(signal=signal, noise=(loop == "noise"),
                     lpf_in_loop=(loop == "lpf"), cost_source=source,
                     integral_clamp=(200.0 if clamp == "clamp" else None))


def make_jax_cost_function(
    params: R2RParameters, line_speed_m_s: float, ti_reference_s: float
):
    """Return ``(scalar_cost, batch_cost)`` backed by the JAX/GPU kernels.

    ``batch_cost`` scores a whole array of candidates in one dispatch, which is
    what makes the 2,805-point grid search cheap; ``scalar_cost`` keeps the same
    signature as the NumPy path so BO can use it unchanged.
    """

    if EVAL_MODEL_OVERRIDE is not None:
        from .retuning_eval_models import ModelEvaluator

        evaluator = ModelEvaluator(params, line_speed_m_s,
                                   _eval_model_from_key(EVAL_MODEL_OVERRIDE))
    else:
        from .retuning_jax import JaxEvaluator

        evaluator = JaxEvaluator.for_campaign(params, line_speed_m_s)

    def batch_cost(kp_values, ti_scales):
        import numpy as _np
        kp = _np.asarray(kp_values, dtype=float)
        ti = _np.asarray(ti_scales, dtype=float) * ti_reference_s
        out = evaluator.cost_only(kp, ti)
        return _np.where(_np.isfinite(out), out, _np.inf)

    def scalar_cost(kp_star: float, ti_scale: float) -> float:
        return float(batch_cost([kp_star], [ti_scale])[0])

    return scalar_cost, batch_cost


def make_cost_function(
    params: R2RParameters, line_speed_m_s: float, auto_ti_s: float
) -> Callable[[float, float], float]:
    """Return ``f(kp_star, ti_scale) -> S`` over the search parameterization.

    ``T_I`` is searched as a multiple of the plant's own ``auto_Ti`` so that one
    box works across a pool spanning three decades of stiffness.
    """

    def cost(kp_star: float, ti_scale: float) -> float:
        try:
            return evaluate_gains(
                params, kp_star, ti_scale * auto_ti_s, line_speed_m_s=line_speed_m_s
            ).S
        except (ValueError, ZeroDivisionError, OverflowError):
            return float("inf")

    return cost


def plant_auto_ti_s(params: R2RParameters, line_speed_m_s: float) -> float:
    """The per-plant integral time implied by theta (the ``auto_Ti`` heuristic)."""

    return auto_tension_integral_time_s(params, line_speed_m_s)


# --------------------------------------------------------------------------- #
# hierarchical grid search on the digital twin
# --------------------------------------------------------------------------- #
def _log_grid(lo: float, hi: float, n: int) -> list[float]:
    """`n` points geometrically spaced over [lo, hi] (both gains span decades)."""

    if n < 2:
        return [math.sqrt(lo * hi)]
    step = (math.log(hi) - math.log(lo)) / (n - 1)
    return [math.exp(math.log(lo) + i * step) for i in range(n)]


@dataclass
class HGSResult:
    """Outcome of the twin-side search.

    ``candidates`` are gain *locations* ordered best-first. Twin costs are kept
    here for diagnostics but must never be handed to a real-plant surrogate.
    """

    best_kp: float
    best_ti_scale: float
    best_twin_cost: float
    candidates: list[tuple[float, float]]
    twin_evaluations: int
    stage_counts: dict[str, int]


def hierarchical_grid_search(
    twin_cost: Callable[[float, float], float],
    *,
    coarse: int = HGS_COARSE,
    lhs: int = HGS_LHS,
    fine: int = HGS_FINE,
    dt_bo: int = HGS_DT_BO,
    seed: int = 0,
    n_candidates: int = 8,
    batch_cost: Callable[[Sequence[float], Sequence[float]], Sequence[float]] | None = None,
) -> HGSResult:
    """Search the twin exhaustively at zero real-plant cost (supplement S8.1).

    Four stages - coarse grid, Latin-hypercube refinement, fine grid around the
    incumbent, then a short BO polish *on the twin*. None of this touches the
    real plant, so none of it counts against the evaluation budget.
    """

    import numpy as np

    scored: list[tuple[float, float, float]] = []  # (cost, kp, ti_scale)
    pending: list[tuple[float, float]] = []

    def score(kp: float, ti: float) -> None:
        # With a batch evaluator the points are queued and dispatched together;
        # that single dispatch is the whole reason the GPU path is fast.
        if batch_cost is not None:
            pending.append((kp, ti))
        else:
            scored.append((twin_cost(kp, ti), kp, ti))

    def flush() -> None:
        if batch_cost is None or not pending:
            return
        kps = [k for k, _ in pending]
        tis = [t for _, t in pending]
        for (kp, ti), value in zip(pending, batch_cost(kps, tis)):
            scored.append((float(value), kp, ti))
        pending.clear()

    # -- stage 1: coarse grid over the whole box --------------------------- #
    side = max(2, int(round(math.sqrt(coarse))))
    for kp in _log_grid(*KP_BOUNDS, side):
        for ti in _log_grid(*TI_SCALE_BOUNDS, side):
            score(kp, ti)
    flush()
    stage_counts = {"coarse": len(scored)}

    # -- stage 2: Latin hypercube over the whole box ----------------------- #
    from scipy.stats import qmc

    sample = qmc.LatinHypercube(d=2, seed=seed).random(n=lhs)
    for u, v in sample:
        kp = math.exp(math.log(KP_BOUNDS[0])
                      + u * (math.log(KP_BOUNDS[1]) - math.log(KP_BOUNDS[0])))
        ti = math.exp(math.log(TI_SCALE_BOUNDS[0])
                      + v * (math.log(TI_SCALE_BOUNDS[1]) - math.log(TI_SCALE_BOUNDS[0])))
        score(kp, ti)
    flush()
    stage_counts["lhs"] = len(scored) - stage_counts["coarse"]

    # -- stage 3: fine grid around the incumbent --------------------------- #
    flush()
    scored.sort(key=lambda t: t[0])
    _, kp0, ti0 = scored[0]
    side = max(2, int(round(math.sqrt(fine))))
    kp_lo = max(KP_BOUNDS[0], kp0 / 3.0)
    kp_hi = min(KP_BOUNDS[1], kp0 * 3.0)
    ti_lo = max(TI_SCALE_BOUNDS[0], ti0 / 3.0)
    ti_hi = min(TI_SCALE_BOUNDS[1], ti0 * 3.0)
    before = len(scored)
    for kp in _log_grid(kp_lo, kp_hi, side):
        for ti in _log_grid(ti_lo, ti_hi, side):
            score(kp, ti)
    flush()
    stage_counts["fine"] = len(scored) - before

    # -- stage 4: short BO polish, still on the twin ----------------------- #
    before = len(scored)
    if dt_bo > 0:
        polished = _gp_minimize_gains(
            lambda kp, ti: twin_cost(kp, ti),
            n_calls=dt_bo + 4,  # gp_minimize needs initial points to fit a GP
            bounds=((kp_lo, kp_hi), (ti_lo, ti_hi)),
            seed=seed,
            x0=[[kp0, ti0]],
        )
        for (kp, ti), cost in polished:
            scored.append((cost, kp, ti))
    stage_counts["dt_bo"] = len(scored) - before

    scored.sort(key=lambda t: t[0])
    best_cost, best_kp, best_ti = scored[0]
    candidates = [(kp, ti) for _, kp, ti in scored[:n_candidates]]
    return HGSResult(best_kp, best_ti, best_cost, candidates, len(scored), stage_counts)


# --------------------------------------------------------------------------- #
# Bayesian optimization on the real plant
# --------------------------------------------------------------------------- #
def _gp_minimize_gains(
    cost: Callable[[float, float], float],
    *,
    n_calls: int,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    seed: int,
    x0: list[list[float]] | None = None,
) -> list[tuple[tuple[float, float], float]]:
    """Run `skopt.gp_minimize` (GP + expected improvement) - supplement S8.1.

    Returns the evaluation trajectory in call order so the caller can form a
    running best-cost curve.
    """

    import warnings

    from skopt import gp_minimize
    from skopt.space import Real

    space = [
        Real(bounds[0][0], bounds[0][1], prior="log-uniform", name="kp"),
        Real(bounds[1][0], bounds[1][1], prior="log-uniform", name="ti"),
    ]

    def objective(x: Sequence[float]) -> float:
        value = cost(float(x[0]), float(x[1]))
        # skopt cannot fit a GP through infinities.
        return float(min(value, 1e6)) if math.isfinite(value) else 1e6

    # skopt rejects a seed point that sits even a float ulp outside the box, and
    # the HGS incumbent lands exactly on a stage-3 boundary often enough to
    # matter. Clamp rather than let the whole cell fail.
    seeded = None
    if x0:
        seeded = [[min(max(pt[0], bounds[0][0]), bounds[0][1]),
                   min(max(pt[1], bounds[1][0]), bounds[1][1])] for pt in x0]
        # skopt evaluates x0 inside the n_calls budget, so the seeds have to be
        # dropped if they would not leave room to sample at all. The real-plant
        # evaluation count IS the claim being tested; it must not overrun.
        seeded = seeded[: max(0, n_calls - 1)] or None

    # n_calls must cover the seeds plus the random initial design.
    n_seeded = len(seeded) if seeded else 0
    n_initial = max(1, min(5, n_calls - n_seeded))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = gp_minimize(
            objective, space, n_calls=n_calls, n_initial_points=n_initial,
            random_state=seed, x0=seeded, acq_func="EI",
        )
    return [((float(x[0]), float(x[1])), float(y))
            for x, y in zip(result.x_iters, result.func_vals)]


# --------------------------------------------------------------------------- #
# identification: build the digital twin of the drifted plant
# --------------------------------------------------------------------------- #
def identify_twin(
    drifted: R2RParameters,
    protocol: Protocol,
    meta: Mapping[str, object],
    *,
    seed: int = 0,
) -> tuple[R2RParameters, dict[str, float]]:
    """Identify the drifted plant under ``protocol`` and rebuild it from theta.

    One excitation record (E_Toggle, 16 s) at the SysID-mode gain, then the v5
    operating-point-weighted PEM. The returned plant is the digital twin: the
    same structure with the seven identified parameters substituted in.
    """

    from ..sysid.estimator import estimate_parameters_weighted_pem
    from .excitations import get_excitation_profile

    line_speed = float(meta.get("v0_mps") or drifted.feeder_velocity_m_s)
    t_max = float(meta.get("T_max_N") or (drifted.tension_ref_N[0] / 0.30))
    v_max = float(meta.get("v_max_mps") or (line_speed / 0.30))

    config = SimulationConfig(
        duration_s=protocol.record_duration_s,
        dt_s=0.001,
        controller_sample_time_s=0.001,
        log_sample_time_s=protocol.log_sample_time_s,
        line_speed_m_s=line_speed,
        sensor_noise_tension_N=protocol.tension_noise_fraction * t_max,
        sensor_noise_velocity_m_s=protocol.velocity_noise_fraction * v_max,
        sensor_lpf_hz=protocol.tension_lpf_hz,
        velocity_lpf_hz=protocol.velocity_lpf_hz,
        noise_rng="numpy_default_rng",
        velocity_seed_offset=100,
        seed=seed,
        controller_tracks_drift=False,
    )
    controller = ControllerConfig(
        target_tension_N=tuple(drifted.tension_ref_N),
        line_speed_m_s=line_speed,
        Kp_star_m_s_per_N=SYSID_MODE_KP,
        TI_s=plant_auto_ti_s(drifted, line_speed),
        paper_velocity_gain_enabled=True,
        high_ea_kp_cap_enabled=False,
        velocity_correction_limit_fraction=None,
    )
    excitation = get_excitation_profile(
        "E_Toggle",
        STEP_FRACTION * float(drifted.tension_ref_N[0]),
        campaign_group="C_retuning_field_matched",
    )
    record = simulate(params=drifted, controller_config=controller, config=config,
                      excitation=excitation, write_output=False)

    fit = estimate_parameters_weighted_pem(
        record.rows, nominal_params=drifted, true_params=drifted
    )
    est = fit.estimates
    twin = replace(
        drifted,
        EA=float(est["EA"]),
        kt_UW=float(est["kt_UW"]), kt_Nip=float(est["kt_Nip"]), kt_RW=float(est["kt_RW"]),
        kf_UW=float(est["kf_UW"]), kf_Nip=float(est["kf_Nip"]), kf_RW=float(est["kf_RW"]),
    )
    return twin, {"mare_theta_percent": 100.0 * float(fit.mare_theta)}


# --------------------------------------------------------------------------- #
# the five retuning strategies
# --------------------------------------------------------------------------- #
@dataclass
class MethodRun:
    """One method's result on one (plant, drift, protocol, seed) cell."""

    method: str
    real_evaluations: int
    final_best_cost: float
    best_kp: float
    best_ti_scale: float
    trajectory: list[float]  # running best cost after each real evaluation
    seed: int
    # Identification quality of the twin this cell's HGS arms transferred from.
    # A transfer failure is only interpretable next to the twin error that
    # caused it, so it travels with every row.
    twin_mare_percent: float | None = None
    # The (kp, ti_scale) points the optimiser actually visited, in call order.
    # The running-best trajectory says how fast a method converged; only this
    # says *where* it walked, which is what a gain-trajectory plot shows.
    gain_path: list[tuple[float, float]] = field(default_factory=list)

    def to_row(self, **extra: object) -> dict[str, object]:
        row = {
            "method": self.method,
            "real_evals": self.real_evaluations,
            "final_best_cost": self.final_best_cost,
            "best_kp": self.best_kp,
            "best_ti_scale": self.best_ti_scale,
            "seed": self.seed,
            "twin_mare_percent": self.twin_mare_percent,
            "gain_path": [list(pt) for pt in self.gain_path],
        }
        row.update(extra)
        return row


def _gain_path(trajectory: Iterable[tuple[tuple[float, float], float]]) -> list[tuple[float, float]]:
    return [(float(pt[0]), float(pt[1])) for pt, _ in trajectory]


def _running_best(trajectory: Iterable[tuple[tuple[float, float], float]]) -> list[float]:
    out: list[float] = []
    best = float("inf")
    for _, value in trajectory:
        best = min(best, value)
        out.append(best)
    return out


def run_cell(
    pool_id: str,
    drift_code: str,
    protocol_name: str,
    *,
    bo_seeds: Sequence[int] = (0, 1, 2),
    hgs_kwargs: Mapping[str, object] | None = None,
    backend: str = "numpy",
) -> list[MethodRun]:
    """Run all five strategies on one plant x drift x protocol cell.

    The stochastic arms (CS-BO, WS-BO) are replicated over ``bo_seeds``; the HGS
    family is deterministic given the twin and runs once, which is the 180/60
    split of Table S9.
    """

    dashboard_id = dict(RETUNING_PLANTS)[pool_id]
    protocol = PROTOCOLS[protocol_name]
    scenario = DRIFT_BY_CODE[drift_code]

    base, meta = parameters_for_plant(dashboard_id)
    line_speed = float(meta.get("v0_mps") or base.feeder_velocity_m_s)
    drifted = apply_drift(base, scenario)

    # One integral-time reference shared by every method and by the twin.
    #
    # T_I is searched as a multiple of this reference, so the reference decides
    # what a given (kp, ti_scale) pair physically *means*. Anchoring the twin to
    # its own auto_Ti and the plant to its own silently rescales T_I on transfer
    # - by up to 8x here, because auto_Ti depends on the identified parameters -
    # so HGS would deliver a different controller than the one it searched. The
    # pre-drift plant's auto_Ti is the honest common anchor: it is known from
    # commissioning, it does not depend on the twin, and it gives all five
    # methods the same absolute search box.
    ti_reference = plant_auto_ti_s(base, line_speed)

    # The twin: identified from one record, and all HGS ever sees.
    twin, fit = identify_twin(drifted, protocol, meta)

    # `backend` swaps the simulator, never the experiment: the JAX kernels are
    # validated against the NumPy path to ~1e-15, so results must not depend on
    # this choice - only the wall clock does.
    twin_batch = None
    if backend == "jax":
        real_cost, _ = make_jax_cost_function(drifted, line_speed, ti_reference)
        twin_cost, twin_batch = make_jax_cost_function(twin, line_speed, ti_reference)
    elif backend == "numpy":
        real_cost = make_cost_function(drifted, line_speed, ti_reference)
        twin_cost = make_cost_function(twin, line_speed, ti_reference)
    else:
        raise ValueError(f"unknown backend {backend!r}; expected 'numpy' or 'jax'")

    runs: list[MethodRun] = []
    bounds = (KP_BOUNDS, TI_SCALE_BOUNDS)

    # -- cold start: no prior model, 30 real evaluations -------------------- #
    for seed in bo_seeds:
        traj = _gp_minimize_gains(real_cost, n_calls=30, bounds=bounds, seed=seed)
        best = min(traj, key=lambda t: t[1])
        runs.append(MethodRun("CS-BO", 30, best[1], best[0][0], best[0][1],
                              _running_best(traj), seed,
                              gain_path=_gain_path(traj)))

    # -- warm start at the SysID-mode operating point ----------------------- #
    # Seeded at K_p* = 100 with T_I from theta_hat - deliberately NOT the twin's
    # optimum, which is the paper's point about why it fails.
    ws_seed_point = [[SYSID_MODE_KP, 1.0]]
    for seed in bo_seeds:
        traj = _gp_minimize_gains(real_cost, n_calls=30, bounds=bounds, seed=seed,
                                  x0=ws_seed_point)
        best = min(traj, key=lambda t: t[1])
        runs.append(MethodRun("WS-BO", 30, best[1], best[0][0], best[0][1],
                              _running_best(traj), seed,
                              gain_path=_gain_path(traj)))

    # -- the HGS family: one twin search, three ways to spend it ------------ #
    hgs = hierarchical_grid_search(
        twin_cost, batch_cost=twin_batch, **(dict(hgs_kwargs or {}))
    )

    # HGS-only: apply the twin's optimum directly, zero real evaluations. The
    # single real run below is the scoring of the delivered gains, not a search
    # step, so it does not count against the budget.
    hgs_only_cost = real_cost(hgs.best_kp, hgs.best_ti_scale)
    runs.append(MethodRun("HGS-only", 0, hgs_only_cost, hgs.best_kp,
                          hgs.best_ti_scale, [hgs_only_cost], 0,
                          gain_path=[(hgs.best_kp, hgs.best_ti_scale)]))

    # HGS+BO(N): refine with N real evaluations seeded from the twin's candidate
    # LOCATIONS. Twin costs are withheld from the surrogate.
    for n_real, label in ((5, "HGS+BO5"), (10, "HGS+BO10")):
        starts = [[kp, ti] for kp, ti in hgs.candidates[:min(3, n_real)]]
        traj = _gp_minimize_gains(real_cost, n_calls=n_real, bounds=bounds,
                                  seed=0, x0=starts)
        # The delivered gain pair is the best of the transfer and the refinement.
        best_point, best_value = min(traj, key=lambda t: t[1])
        if hgs_only_cost <= best_value:
            best_point, best_value = (hgs.best_kp, hgs.best_ti_scale), hgs_only_cost
        running = [min(hgs_only_cost, v) for v in _running_best(traj)]
        runs.append(MethodRun(label, n_real, best_value, best_point[0],
                              best_point[1], running, 0,
                              gain_path=_gain_path(traj)))

    for run in runs:
        run.twin_mare_percent = fit["mare_theta_percent"]
    return runs
