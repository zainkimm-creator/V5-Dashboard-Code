"""One-step prediction-error system identification for the seven paper parameters.

The v5 canonical estimator is `estimate_parameters_weighted_pem`. It minimizes an
operating-point-weighted one-step prediction error over the full six-channel
logged state, which is what the paper Eq. (7) cost now specifies:

    J(theta) = sum_k || W (z_k - zhat_k(theta)) ||^2

`z_k` is the logged state measurement (three span tensions and three roller
angular velocities), deliberately distinct from the controlled output
`y = [T1, T2, T3]`. `W` divides each tension residual by its set-point tension
and each velocity residual by its steady-state angular velocity, so both
channels land on a common dimensionless percent-of-operating-point scale.

The two earlier estimators are kept below for comparison views only. They fit
the tension channel with an identity weight, which is what v4.1 specified; their
accuracy numbers are not interchangeable with the weighted ones.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from backend.models.equations import (
    INPUT_NAMES,
    PARAMETER_NAMES,
    R2RParameters,
    STATE_NAMES,
    roller_tension_differences,
    steady_surface_velocities,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY_DIR = PROJECT_ROOT / "reports" / "validation_summary"

LOGGED_STATE_NAMES = ("T1", "T2", "T3", "omega_UW", "omega_Nip", "omega_RW")
WEIGHT_CHANNEL_NAMES = (
    "1/T_ref_1",
    "1/T_ref_2",
    "1/T_ref_3",
    "1/omega_ss_UW",
    "1/omega_ss_Nip",
    "1/omega_ss_RW",
)

# v5 canonical initialization: the true parameters scaled by 1.01, with the
# unwinder and rewinder entries first symmetrized to their common mean, box
# constrained to one decade either side. This is fixed, not an experimental axis.
CANONICAL_INITIAL_SCALE = 1.01
CANONICAL_BOUND_DECADE = 10.0

# Reel-symmetric parameter pairs, by index into PARAMETER_NAMES.
_REEL_PAIRS = ((0, 2), (3, 5))  # (kt_UW, kt_RW) and (kf_UW, kf_RW)


@dataclass
class SysIDResult:
    estimates: dict[str, float]
    mare_theta: float
    error_table: list[dict[str, float | str]]
    summary_path: str | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "estimates": self.estimates,
            "MARE_theta": self.mare_theta,
            "metric_name": "MARE_theta",
            "error_table": self.error_table,
            "summary_path": self.summary_path,
            "diagnostics": self.diagnostics,
        }


def load_rows_from_csv(path: str | Path) -> list[dict[str, float]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [{key: float(value) for key, value in row.items()} for row in reader]


def _solve_2x2(a11: float, a12: float, a22: float, b1: float, b2: float) -> tuple[float, float]:
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-12:
        raise ValueError("ill-conditioned 2x2 normal equation")
    return ((b1 * a22 - b2 * a12) / det, (a11 * b2 - a12 * b1) / det)


def _safe_relative_error(estimate: float, truth: float) -> float:
    denom = abs(truth) if abs(truth) > 1e-12 else 1.0
    return (estimate - truth) / denom


def _time_steps(rows: Sequence[Mapping[str, float]]) -> list[float]:
    return [rows[i + 1]["time_s"] - rows[i]["time_s"] for i in range(len(rows) - 1)]


def operating_point_weights(
    params: R2RParameters,
    line_speed_m_s: float | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Return the diagonal of `W` for one measurement condition.

    `W = diag(1/T_ref,1, 1/T_ref,2, 1/T_ref,3, 1/omega_ss,UW, 1/omega_ss,Nip,
    1/omega_ss,RW)`. The tension weights use the controller set-point `T_ref`,
    not the load-cell full scale `T_max` that defines the noise model. The
    velocity weights use the exact equilibrium `omega_ss`, not the `v0/R`
    approximation. Both are defined in the noise-free case too, so the weighting
    is parameter-free and needs no noise model.

    Each measurement condition carries its own `W`, because multi-speed
    excitations sit at different steady states.
    """

    speed = float(line_speed_m_s if line_speed_m_s is not None else params.feeder_velocity_m_s)
    steady_velocities = steady_surface_velocities(params, speed)
    omega_ss = tuple(
        velocity / radius
        for velocity, radius in zip(steady_velocities, params.roller_radius_m, strict=True)
    )
    channels = (*params.tension_ref_N, *omega_ss)
    weights = []
    for value in channels:
        magnitude = abs(float(value))
        if magnitude < 1e-12 or not math.isfinite(magnitude):
            raise ValueError(
                "operating-point weighting needs a non-zero finite T_ref and omega_ss "
                "on every logged channel"
            )
        weights.append(1.0 / magnitude)
    return tuple(weights)  # type: ignore[return-value]


def canonical_initial_theta(
    reference: Sequence[float],
    *,
    initial_scale: float = CANONICAL_INITIAL_SCALE,
    symmetrize_reels: bool = True,
) -> list[float]:
    """Return the fixed v5 starting point for identification.

    The unwinder and rewinder entries are first symmetrized to their common
    mean, then the whole vector is scaled by `initial_scale`. The starting point
    is fixed and is not an experimental axis; there is no `alpha` sweep.
    """

    theta = [float(value) for value in reference]
    if len(theta) != len(PARAMETER_NAMES):
        raise ValueError(f"reference theta must contain {len(PARAMETER_NAMES)} values")
    if symmetrize_reels:
        for low, high in _REEL_PAIRS:
            common_mean = 0.5 * (theta[low] + theta[high])
            theta[low] = common_mean
            theta[high] = common_mean
    return [value * float(initial_scale) for value in theta]


def estimate_parameters(
    rows: Sequence[Mapping[str, float]],
    nominal_params: R2RParameters | None = None,
    true_params: R2RParameters | None = None,
    summary_name: str | None = "sysid_result.json",
    summary_dir: Path | None = None,
) -> SysIDResult:
    """Estimate `kt_UW`, `kt_Nip`, `kt_RW`, `kf_UW`, `kf_Nip`, `kf_RW`, and `EA`.

    The estimator uses one-step finite-difference prediction equations. For roller
    dynamics, each roller solves a two-parameter least-squares problem in the
    paper Eq. (6) ratio form:

        dv_i = kt_i*(topology_tension_term_i + u_i/R_i) - kf_i*v_i

    For tension dynamics, all three spans share one least-squares estimate of EA
    derived from paper Eq. (1):

        dT_i - (T_{i-1}v_{i-1} - T_i v_i)/L_i = EA*(v_i - v_{i-1})/L_i
    """

    if len(rows) < 3:
        raise ValueError("at least 3 logged rows are required for SysID")
    params = nominal_params or R2RParameters()
    true = true_params or params
    dt_values = _time_steps(rows)
    if any(dt <= 0 for dt in dt_values):
        raise ValueError("row time_s values must be strictly increasing")

    kt_estimates: list[float] = []
    kf_estimates: list[float] = []
    for roller_idx in range(3):
        a11 = a12 = a22 = b1 = b2 = 0.0
        input_name = INPUT_NAMES[roller_idx]
        radius = params.roller_radius_m[roller_idx]
        velocity_name = ("v_UW_m_s", "v_Nip_m_s", "v_RW_m_s")[roller_idx]
        for i in range(len(rows) - 1):
            dt = dt_values[i]
            row = rows[i]
            next_row = rows[i + 1]
            velocity = row[velocity_name]
            dv = (next_row[velocity_name] - velocity) / dt
            state = tuple(float(row[name]) for name in STATE_NAMES)
            tension_delta = roller_tension_differences(state)[roller_idx]
            y = dv
            x1 = tension_delta + (row[input_name] / radius)
            x2 = -velocity
            a11 += x1 * x1
            a12 += x1 * x2
            a22 += x2 * x2
            b1 += x1 * y
            b2 += x2 * y
        kt_i, kf_i = _solve_2x2(a11, a12, a22, b1, b2)
        kt_estimates.append(max(1e-6, kt_i))
        kf_estimates.append(max(1e-6, kf_i))

    ea_num = 0.0
    ea_den = 0.0
    def feeder_speed(row: Mapping[str, float]) -> float:
        return float(row.get("line_speed_ref_m_s", params.feeder_velocity_m_s))

    span_rows = (
        ("T1", 0, lambda row: (0.0, row["T1"]), lambda row: (row["v_UW_m_s"], feeder_speed(row))),
        ("T2", 1, lambda row: (row["T1"], row["T2"]), lambda row: (feeder_speed(row), row["v_Nip_m_s"])),
        ("T3", 2, lambda row: (row["T2"], row["T3"]), lambda row: (row["v_Nip_m_s"], row["v_RW_m_s"])),
    )
    for i in range(len(rows) - 1):
        dt = dt_values[i]
        row = rows[i]
        next_row = rows[i + 1]
        for tension_name, span_idx, tension_pair_fn, speed_pair_fn in span_rows:
            d_tension = (next_row[tension_name] - row[tension_name]) / dt
            t_prev, t_i = tension_pair_fn(row)
            v_prev, v_i = speed_pair_fn(row)
            length = params.span_length_m[span_idx]
            convective = (t_prev * v_prev - t_i * v_i) / length
            y = d_tension - convective
            x = (v_i - v_prev) / length
            ea_num += x * y
            ea_den += x * x
    ea_estimate = ea_num / ea_den if ea_den > 1e-12 else params.EA

    estimates = {
        "kt_UW": kt_estimates[0],
        "kt_Nip": kt_estimates[1],
        "kt_RW": kt_estimates[2],
        "kf_UW": kf_estimates[0],
        "kf_Nip": kf_estimates[1],
        "kf_RW": kf_estimates[2],
        "EA": max(1e-6, ea_estimate),
    }
    error_table: list[dict[str, float | str]] = []
    rel_errors = []
    truth_values = true.sysid_values()
    for name in PARAMETER_NAMES:
        estimate = estimates[name]
        truth = truth_values[name]
        abs_error = estimate - truth
        rel_error = _safe_relative_error(estimate, truth)
        rel_errors.append(rel_error)
        error_table.append(
            {
                "parameter": name,
                "estimate": estimate,
                "truth": truth,
                "absolute_error": abs_error,
                "relative_error": rel_error,
            }
        )
    mare_theta = sum(abs(value) for value in rel_errors) / len(rel_errors)

    summary_path = None
    if summary_name:
        target_dir = summary_dir or DEFAULT_SUMMARY_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / summary_name
        target_path.write_text(
            json.dumps(
                {
                    "estimates": estimates,
                    "MARE_theta": mare_theta,
                    "metric_name": "MARE_theta",
                    "metric_formula": "mean(abs((theta_hat-theta_true)/theta_true))",
                    "error_table": error_table,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        summary_path = str(target_path)
    return SysIDResult(
        estimates=estimates,
        mare_theta=mare_theta,
        error_table=error_table,
        summary_path=summary_path,
        diagnostics={
            "objective": "one_step_finite_difference_ratio_least_squares",
            "optimizer": "closed_form_normal_equations",
            "success": True,
            "parameter_count": len(PARAMETER_NAMES),
            "sample_count": len(rows),
        },
    )


@dataclass(frozen=True)
class MeasurementCondition:
    """One logged acquisition that takes part in a joint fit.

    A single-condition fit holds one of these. `ET3M` and the two-operating-point
    excitation hold several: their operating points are logged as separate
    experiments and identified in one joint fit through the multi-condition
    cost, not averaged over separate per-operating-point fits. Each condition
    carries its own `W`, since the steady state differs between them.
    """

    rows: Sequence[Mapping[str, float]]
    line_speed_m_s: float | None = None
    params: R2RParameters | None = None
    label: str = "c1"


def _condition_arrays(
    condition: MeasurementCondition,
    nominal: R2RParameters,
    break_on_line_speed_change: bool,
):
    """Build the one-step pair arrays and the weight vector for one condition."""

    import numpy as np

    rows = condition.rows
    if len(rows) < 3:
        raise ValueError("at least 3 logged rows are required for SysID")
    params = condition.params or nominal
    states = np.asarray([[row[name] for name in LOGGED_STATE_NAMES] for row in rows], dtype=float)
    inputs = np.asarray([[row[name] for name in INPUT_NAMES] for row in rows[:-1]], dtype=float)
    all_line_speed = np.asarray(
        [row.get("line_speed_ref_m_s", params.feeder_velocity_m_s) for row in rows],
        dtype=float,
    )
    times = np.asarray([row["time_s"] for row in rows], dtype=float)
    dt = np.diff(times)
    if np.any(dt <= 0.0):
        raise ValueError("row time_s values must be strictly increasing")

    pair_mask = np.ones(len(rows) - 1, dtype=bool)
    if break_on_line_speed_change:
        pair_mask &= np.isclose(all_line_speed[:-1], all_line_speed[1:], rtol=0.0, atol=1e-12)
        if not np.any(pair_mask):
            raise ValueError("no within-segment one-step pairs remain after the line-speed split")

    weight_speed = condition.line_speed_m_s
    if weight_speed is None:
        # The steady state this condition sits at, taken from the logged run.
        weight_speed = float(np.median(all_line_speed))
    weights = np.asarray(operating_point_weights(params, weight_speed), dtype=float)

    return {
        "label": condition.label,
        "current": states[:-1][pair_mask],
        "measured_next": states[1:][pair_mask],
        "inputs": inputs[pair_mask],
        "line_speed": all_line_speed[:-1][pair_mask],
        "dt": dt[pair_mask],
        "radii": np.asarray(params.roller_radius_m, dtype=float),
        "lengths": np.asarray(params.span_length_m, dtype=float),
        "weights": weights,
        "weight_line_speed_m_s": float(weight_speed),
        "pair_count": int(pair_mask.sum()),
    }


def estimate_parameters_weighted_pem(
    rows: Sequence[Mapping[str, float]] | Sequence[MeasurementCondition],
    nominal_params: R2RParameters | None = None,
    true_params: R2RParameters | None = None,
    *,
    initial_scale: float = CANONICAL_INITIAL_SCALE,
    bound_decade: float = CANONICAL_BOUND_DECADE,
    symmetrize_reels: bool = True,
    max_nfev: int = 150,
    break_on_line_speed_change: bool = False,
) -> SysIDResult:
    """Estimate the seven parameters with the v5 operating-point-weighted PEM.

    Minimizes `sum_k || W (z_k - zhat_k(theta)) ||^2` over the six-channel logged
    state. Each logged state is advanced one logging interval with RK4 and
    `scipy.optimize.least_squares` (trust-region reflective) minimizes the
    weighted residual. There is no Nelder-Mead branch.

    The starting point is fixed at `1.01 * theta_true` with the unwinder and
    rewinder entries symmetrized to their common mean, box constrained to one
    decade either side. The optimizer never reads `true_params` beyond that
    published initialization rule; ground truth is otherwise used only after
    estimation, to compute MARE.
    """

    import numpy as np
    from scipy.optimize import least_squares

    nominal = nominal_params or R2RParameters()
    true = true_params or nominal
    if bound_decade <= 1.0 or not math.isfinite(bound_decade):
        raise ValueError("bound_decade must be finite and greater than 1")
    if initial_scale <= 0.0 or not math.isfinite(initial_scale):
        raise ValueError("initial_scale must be finite and positive")

    if rows and isinstance(rows[0], MeasurementCondition):
        conditions = list(rows)  # type: ignore[arg-type]
    else:
        conditions = [MeasurementCondition(rows=rows)]  # type: ignore[arg-type]
    if not conditions:
        raise ValueError("at least one measurement condition is required")

    prepared = [
        _condition_arrays(condition, nominal, break_on_line_speed_change)
        for condition in conditions
    ]

    theta_init = np.asarray(
        canonical_initial_theta(
            [true.sysid_values()[name] for name in PARAMETER_NAMES],
            initial_scale=initial_scale,
            symmetrize_reels=symmetrize_reels,
        ),
        dtype=float,
    )
    if np.any(theta_init <= 0.0):
        raise ValueError("the initial theta must be strictly positive to use log-ratio scaling")

    def rhs(state, theta, block):
        kt = theta[:3]
        kf = theta[3:6]
        ea = theta[6]
        radii = block["radii"]
        lengths = block["lengths"]
        line_speed = block["line_speed"]
        t1, t2, t3 = state[:, :3].T
        v_uw, v_nip, v_rw = (state[:, 3:] * radii).T
        d_tension = np.column_stack(
            (
                (ea * (line_speed - v_uw) - line_speed * t1) / lengths[0],
                (ea * (v_nip - line_speed) + t1 * line_speed - t2 * v_nip) / lengths[1],
                (ea * (v_rw - v_nip) + t2 * v_nip - t3 * v_rw) / lengths[2],
            )
        )
        tension_delta = np.column_stack((-t1, t2 - t3, t3))
        d_velocity = kt * (tension_delta + block["inputs"] / radii) - kf * np.column_stack(
            (v_uw, v_nip, v_rw)
        )
        return np.column_stack((d_tension, d_velocity / radii))

    def predict(theta, block):
        current = block["current"]
        h = block["dt"][:, None]
        k1 = rhs(current, theta, block)
        k2 = rhs(current + 0.5 * h * k1, theta, block)
        k3 = rhs(current + 0.5 * h * k2, theta, block)
        k4 = rhs(current + h * k3, theta, block)
        return current + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def residual(log_ratio):
        theta = theta_init * np.exp(log_ratio)
        blocks = []
        for block in prepared:
            weighted = (predict(theta, block) - block["measured_next"]) * block["weights"]
            blocks.append(weighted.ravel())
        values = np.concatenate(blocks)
        if not np.all(np.isfinite(values)):
            return np.full(values.shape, 1e6)
        return values

    log_bound = math.log(bound_decade)
    solution = least_squares(
        residual,
        np.zeros(7),
        bounds=(np.full(7, -log_bound), np.full(7, log_bound)),
        method="trf",
        max_nfev=max_nfev,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
    )
    estimate_values = theta_init * np.exp(solution.x)
    estimates = dict(zip(PARAMETER_NAMES, (float(value) for value in estimate_values), strict=True))

    # Fisher information on the weighted residual scale: F = sum_k J_k^T W^2 J_k.
    # `solution.jac` is already the Jacobian of the weighted residual, but with
    # respect to the log-ratio; the chain rule divides column j by theta_j to put
    # it back in parameter space.
    fisher_ranking_basis = None
    log10_kappa_fisher = None
    # Reported alongside the existing ranking basis for the full-sweep export.
    # `F` is symmetric positive semi-definite, so its singular values are its
    # eigenvalues; cond / logdet / lambda_min are read straight off them and add
    # no computation to the fit itself.
    fisher_cond = None
    fisher_logdet = None
    fisher_lambda_min = None
    try:
        jac_theta = np.asarray(solution.jac, dtype=float) / estimate_values[None, :]
        fisher = jac_theta.T @ jac_theta
        singular_values = np.linalg.svd(fisher, compute_uv=False)
        if singular_values[-1] > 0.0 and np.all(np.isfinite(singular_values)):
            log10_kappa_fisher = float(np.log10(singular_values[0] / singular_values[-1]))
            fisher_ranking_basis = "log10_condition_number_of_weighted_fisher_information"
            fisher_cond = float(singular_values[0] / singular_values[-1])
            fisher_logdet = float(np.sum(np.log(singular_values)))
            fisher_lambda_min = float(singular_values[-1])
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        log10_kappa_fisher = None

    parameter_ratios = estimate_values / theta_init
    lower_bound_parameters = [
        name
        for name, ratio in zip(PARAMETER_NAMES, parameter_ratios, strict=True)
        if ratio <= (1.0 / bound_decade) * 1.0001
    ]
    upper_bound_parameters = [
        name
        for name, ratio in zip(PARAMETER_NAMES, parameter_ratios, strict=True)
        if ratio >= bound_decade * 0.9999
    ]

    truth_values = true.sysid_values()
    error_table: list[dict[str, float | str]] = []
    relative_errors: list[float] = []
    for name in PARAMETER_NAMES:
        estimate = estimates[name]
        truth = truth_values[name]
        relative_error = _safe_relative_error(estimate, truth)
        relative_errors.append(relative_error)
        error_table.append(
            {
                "parameter": name,
                "estimate": estimate,
                "truth": truth,
                "absolute_error": estimate - truth,
                "relative_error": relative_error,
            }
        )

    return SysIDResult(
        estimates=estimates,
        mare_theta=sum(abs(value) for value in relative_errors) / len(relative_errors),
        error_table=error_table,
        summary_path=None,
        diagnostics={
            "objective": "paper_eq7_weighted_one_step_prediction_error",
            "residual_channels": list(LOGGED_STATE_NAMES),
            "weight_channels": list(WEIGHT_CHANNEL_NAMES),
            "weight_definition": (
                "W = diag(1/T_ref_1, 1/T_ref_2, 1/T_ref_3, "
                "1/omega_ss_UW, 1/omega_ss_Nip, 1/omega_ss_RW)"
            ),
            "condition_count": len(prepared),
            "conditions": [
                {
                    "label": block["label"],
                    "pair_count": block["pair_count"],
                    "weight_line_speed_m_s": block["weight_line_speed_m_s"],
                    "weights": [float(value) for value in block["weights"]],
                }
                for block in prepared
            ],
            "joint_fit": len(prepared) > 1,
            "optimizer": "scipy_least_squares_trf",
            "initial_scale": float(initial_scale),
            "initial_theta": {
                name: float(value)
                for name, value in zip(PARAMETER_NAMES, theta_init, strict=True)
            },
            "initialization_rule": (
                "fixed at initial_scale * theta_true with UW/RW symmetrized to their "
                "common mean; not an experimental axis"
            ),
            "reels_symmetrized": bool(symmetrize_reels),
            "bound_decade": float(bound_decade),
            "max_nfev": int(max_nfev),
            "success": bool(solution.success),
            "status": int(solution.status),
            "message": str(solution.message),
            "nfev": int(solution.nfev),
            "njev": None if solution.njev is None else int(solution.njev),
            "cost": float(solution.cost),
            "optimality": float(solution.optimality),
            "residual_count": int(solution.fun.size),
            "weighted_residual_rmse": float(np.sqrt(np.mean(np.square(solution.fun)))),
            "active_mask": [int(value) for value in solution.active_mask],
            "parameter_ratios_to_initial": {
                name: float(value)
                for name, value in zip(PARAMETER_NAMES, parameter_ratios, strict=True)
            },
            "lower_bound_parameters": lower_bound_parameters,
            "upper_bound_parameters": upper_bound_parameters,
            "log10_kappa_fisher": log10_kappa_fisher,
            "fisher_cond": fisher_cond,
            "fisher_logdet": fisher_logdet,
            "fisher_lambda_min": fisher_lambda_min,
            "fisher_information": "F = sum_k J_k^T W^2 J_k",
            "fisher_ranking_basis": fisher_ranking_basis,
            "fisher_reporting_note": (
                "On the weighted-residual scale the absolute value of kappa(F) depends "
                "on the operating-point normalization, so only the ordering across "
                "plants or excitations is interpretable. Rank, do not quote."
            ),
        },
    )


def estimate_parameters_one_step_pem(
    rows: Sequence[Mapping[str, float]],
    nominal_params: R2RParameters | None = None,
    true_params: R2RParameters | None = None,
    *,
    initial_scale: float = 1.5,
    lower_scale: float = 0.1,
    upper_scale: float = 10.0,
    max_nfev: int = 150,
    break_on_line_speed_change: bool = False,
) -> SysIDResult:
    """Estimate the seven parameters from nonlinear one-step tension residuals.

    This implements the paper Eq. (7) objective. Each logged state is advanced
    one logging interval with RK4, and SciPy TRF minimizes the three predicted
    tension residuals. The optimizer never uses ``true_params``; ground truth is
    supplied only after estimation to calculate MARE for validation.
    """

    if len(rows) < 3:
        raise ValueError("at least 3 logged rows are required for SysID")
    if initial_scale <= 0.0 or not math.isfinite(initial_scale):
        raise ValueError("initial_scale must be finite and positive")
    if lower_scale <= 0.0 or upper_scale <= lower_scale:
        raise ValueError("parameter scale bounds must be positive and increasing")

    import numpy as np
    from scipy.optimize import least_squares

    nominal = nominal_params or R2RParameters()
    true = true_params or nominal
    state_names = ("T1", "T2", "T3", "omega_UW", "omega_Nip", "omega_RW")
    states = np.asarray([[row[name] for name in state_names] for row in rows], dtype=float)
    inputs = np.asarray([[row[name] for name in INPUT_NAMES] for row in rows[:-1]], dtype=float)
    line_speed = np.asarray(
        [row.get("line_speed_ref_m_s", nominal.feeder_velocity_m_s) for row in rows[:-1]],
        dtype=float,
    )
    times = np.asarray([row["time_s"] for row in rows], dtype=float)
    dt = np.diff(times)
    if np.any(dt <= 0.0):
        raise ValueError("row time_s values must be strictly increasing")

    pair_mask = np.ones(len(rows) - 1, dtype=bool)
    if break_on_line_speed_change:
        all_line_speed = np.asarray(
            [row.get("line_speed_ref_m_s", nominal.feeder_velocity_m_s) for row in rows],
            dtype=float,
        )
        pair_mask &= np.isclose(
            all_line_speed[:-1],
            all_line_speed[1:],
            rtol=0.0,
            atol=1e-12,
        )
        if not np.any(pair_mask):
            raise ValueError("no within-segment one-step pairs remain after the line-speed split")

    current = states[:-1][pair_mask]
    measured_next_tension = states[1:, :3][pair_mask]
    radii = np.asarray(nominal.roller_radius_m, dtype=float)
    lengths = np.asarray(nominal.span_length_m, dtype=float)
    nominal_theta = np.asarray([*nominal.paper_kt, *nominal.paper_kf, nominal.EA], dtype=float)
    inputs = inputs[pair_mask]
    line_speed = line_speed[pair_mask]
    dt = dt[pair_mask]

    def rhs(state, theta):
        kt = theta[:3]
        kf = theta[3:6]
        ea = theta[6]
        t1, t2, t3 = state[:, :3].T
        v_uw, v_nip, v_rw = (state[:, 3:] * radii).T
        d_tension = np.column_stack(
            (
                (ea * (line_speed - v_uw) - line_speed * t1) / lengths[0],
                (ea * (v_nip - line_speed) + t1 * line_speed - t2 * v_nip) / lengths[1],
                (ea * (v_rw - v_nip) + t2 * v_nip - t3 * v_rw) / lengths[2],
            )
        )
        tension_delta = np.column_stack((-t1, t2 - t3, t3))
        d_velocity = kt * (tension_delta + inputs / radii) - kf * np.column_stack((v_uw, v_nip, v_rw))
        return np.column_stack((d_tension, d_velocity / radii))

    def predict(theta):
        h = dt[:, None]
        k1 = rhs(current, theta)
        k2 = rhs(current + 0.5 * h * k1, theta)
        k3 = rhs(current + 0.5 * h * k2, theta)
        k4 = rhs(current + h * k3, theta)
        return current + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    tension_scale = max(float(np.std(measured_next_tension)), 1e-6)

    def residual(log_ratio):
        theta = nominal_theta * np.exp(log_ratio)
        values = ((predict(theta)[:, :3] - measured_next_tension) / tension_scale).ravel()
        if not np.all(np.isfinite(values)):
            return np.full(values.shape, 1e6)
        return values

    solution = least_squares(
        residual,
        np.full(7, math.log(initial_scale)),
        bounds=(np.full(7, math.log(lower_scale)), np.full(7, math.log(upper_scale))),
        method="trf",
        max_nfev=max_nfev,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
    )
    estimate_values = nominal_theta * np.exp(solution.x)
    estimates = dict(zip(PARAMETER_NAMES, (float(value) for value in estimate_values), strict=True))
    parameter_ratios = estimate_values / nominal_theta
    lower_bound_parameters = [
        name
        for name, ratio in zip(PARAMETER_NAMES, parameter_ratios, strict=True)
        if ratio <= lower_scale * 1.0001
    ]
    upper_bound_parameters = [
        name
        for name, ratio in zip(PARAMETER_NAMES, parameter_ratios, strict=True)
        if ratio >= upper_scale * 0.9999
    ]
    truth_values = true.sysid_values()
    error_table: list[dict[str, float | str]] = []
    relative_errors: list[float] = []
    for name in PARAMETER_NAMES:
        estimate = estimates[name]
        truth = truth_values[name]
        relative_error = _safe_relative_error(estimate, truth)
        relative_errors.append(relative_error)
        error_table.append(
            {
                "parameter": name,
                "estimate": estimate,
                "truth": truth,
                "absolute_error": estimate - truth,
                "relative_error": relative_error,
            }
        )
    return SysIDResult(
        estimates=estimates,
        mare_theta=sum(abs(value) for value in relative_errors) / len(relative_errors),
        error_table=error_table,
        summary_path=None,
        diagnostics={
            "objective": "paper_eq7_one_step_tension_prediction_error",
            "optimizer": "scipy_least_squares_trf",
            "initial_scale": float(initial_scale),
            "lower_scale": float(lower_scale),
            "upper_scale": float(upper_scale),
            "max_nfev": int(max_nfev),
            "success": bool(solution.success),
            "status": int(solution.status),
            "message": str(solution.message),
            "nfev": int(solution.nfev),
            "njev": None if solution.njev is None else int(solution.njev),
            "cost": float(solution.cost),
            "optimality": float(solution.optimality),
            "residual_count": int(solution.fun.size),
            "normalized_residual_rmse": float(np.sqrt(np.mean(np.square(solution.fun)))),
            "active_mask": [int(value) for value in solution.active_mask],
            "parameter_ratios_to_nominal": {
                name: float(value)
                for name, value in zip(PARAMETER_NAMES, parameter_ratios, strict=True)
            },
            "lower_bound_parameters": lower_bound_parameters,
            "upper_bound_parameters": upper_bound_parameters,
        },
    )
