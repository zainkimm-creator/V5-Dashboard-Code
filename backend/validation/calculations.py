"""Worked calculation payloads for frontend result panels."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from backend.models.controller import ControllerConfig
from backend.models.equations import INPUT_NAMES, R2RParameters


def _clean_number(value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        return number
    return round(number, 8)


def _format_number(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        return str(number)
    return f"{number:.6g}"


def _safe_relative(value: float, reference: float) -> float:
    if abs(reference) < 1e-12:
        return 0.0 if abs(value) < 1e-12 else math.inf
    return (value - reference) / reference


def _calculation(
    *,
    title: str,
    parameter: str,
    formula: str,
    values: Mapping[str, object],
    substitution: str,
    result: str,
    summary: str,
    steps: Sequence[str] | None = None,
) -> dict[str, object]:
    return {
        "title": title,
        "parameter": parameter,
        "formula": formula,
        "values": dict(values),
        "substitution": substitution,
        "result": result,
        "summary": summary,
        "steps": list(steps or []),
    }


def simulation_calculation_payload(
    metrics: Mapping[str, float],
    rows: Sequence[Mapping[str, float]],
    config: object,
    params: R2RParameters,
) -> dict[str, object]:
    """Return worked calculations for closed-loop simulation metrics."""

    if not rows:
        return {"calculation_summary": "No rows were available for calculations.", "calculations": []}

    tension_names = ("T1", "T2", "T3")
    target = ControllerConfig(line_speed_m_s=getattr(config, "line_speed_m_s", 1.0)).target_tension_N
    squared_errors = [
        (float(row[name]) - target[index]) ** 2
        for row in rows
        for index, name in enumerate(tension_names)
    ]
    sse = sum(squared_errors)
    tension_terms = len(squared_errors)
    tension_tracking_error = math.sqrt(sse / tension_terms)

    effort_squares = [float(row[name]) ** 2 for row in rows for name in INPUT_NAMES]
    effort_sse = sum(effort_squares)
    effort_terms = len(effort_squares)
    effort_rms = math.sqrt(effort_sse / effort_terms)

    derivative_row = max(
        rows,
        key=lambda row: abs(float(row["v_UW_m_s"]) - params.feeder_velocity_m_s)
        + abs(float(row["T1"]) * float(row["v_UW_m_s"])),
    )
    v_prev = float(derivative_row["v_UW_m_s"])
    v_i = params.feeder_velocity_m_s
    t_prev = 0.0
    t_i = float(derivative_row["T1"])
    speed_delta = v_i - v_prev
    elastic_term = (params.EA / params.span_length_m[0]) * speed_delta
    convective_term = (t_prev * v_prev - t_i * v_i) / params.span_length_m[0]
    d_t1 = elastic_term + convective_term

    calculations = [
        _calculation(
            title="Tension Tracking Error",
            parameter="tension_tracking_error_N",
            formula="sqrt(sum((T_i - T_ref_i)^2) / (3*N))",
            values={
                "logged_rows": float(len(rows)),
                "terms": float(tension_terms),
                "target_T1_N": target[0],
                "target_T2_N": target[1],
                "target_T3_N": target[2],
                "squared_error_sum_N2": _clean_number(sse),
            },
            substitution=f"sqrt({_format_number(sse)} / {tension_terms})",
            result=f"tension_tracking_error_N = {_format_number(tension_tracking_error)} N",
            summary="This measures average tension tracking error across all three spans and all logged samples.",
            steps=[
                f"Use {len(rows)} logged rows from the simulation output.",
                f"For each row, compare T1, T2, and T3 with target tensions {tuple(_format_number(v) for v in target)} N.",
                "Square every tension error so positive and negative tracking errors both count.",
                f"Add all squared errors: sum(error^2) = {_format_number(sse)} N^2.",
                f"Divide by the number of tension samples: {tension_terms}.",
                f"Take the square root to get tension_tracking_error_N = {_format_number(tension_tracking_error)} N.",
            ],
        ),
        _calculation(
            title="Control Effort RMS",
            parameter="control_effort_rms_V",
            formula="sqrt(sum(u_UW^2 + u_Nip^2 + u_RW^2) / (3*N))",
            values={
                "logged_rows": float(len(rows)),
                "terms": float(effort_terms),
                "input_square_sum_V2": _clean_number(effort_sse),
            },
            substitution=f"sqrt({_format_number(effort_sse)} / {effort_terms})",
            result=f"control_effort_rms_V = {_format_number(effort_rms)} V",
            summary="This shows how much voltage effort the controller used while regulating the web.",
            steps=[
                f"Read u_UW, u_Nip, and u_RW from every one of the {len(rows)} logged rows.",
                "Square each voltage command so positive and negative effort contribute equally.",
                f"Add all squared voltage commands: sum(u^2) = {_format_number(effort_sse)} V^2.",
                f"Divide by the number of command samples: {effort_terms}.",
                f"Take the square root to get control_effort_rms_V = {_format_number(effort_rms)} V.",
            ],
        ),
        _calculation(
            title="T1 Derivative Example",
            parameter="dT1/dt",
            formula="(EA/L1)*(v1 - v0) + (T0*v0 - T1*v1)/L1",
            values={
                "time_s": _clean_number(float(derivative_row["time_s"])),
                "EA_N": params.EA,
                "L1_m": params.span_length_m[0],
                "T0_N": t_prev,
                "T1_N": _clean_number(float(derivative_row["T1"])),
                "v0_UW_m_s": _clean_number(v_prev),
                "v1_feeder_m_s": _clean_number(v_i),
                "elastic_term_N_per_s": _clean_number(elastic_term),
                "convective_term_N_per_s": _clean_number(convective_term),
            },
            substitution=(
                f"({params.EA}/{params.span_length_m[0]})*({_format_number(speed_delta)}) "
                f"+ (({_format_number(t_prev)}*{_format_number(v_prev)}) "
                f"- ({_format_number(t_i)}*{_format_number(v_i)}))/{params.span_length_m[0]}"
            ),
            result=f"dT1/dt = {_format_number(d_t1)} N/s",
            summary="The tension equation combines elastic stretch from velocity mismatch with convective transport of tension across the span.",
            steps=[
                f"Pick the logged row at time { _format_number(float(derivative_row['time_s'])) } s as a worked example.",
                f"Use the boundary tension T0 = {_format_number(t_prev)} N and upstream roller speed v0 = v_UW = {_format_number(v_prev)} m/s.",
                f"Read T1 = {_format_number(t_i)} N and feeder speed v1 = {_format_number(v_i)} m/s.",
                f"Compute velocity mismatch: v1 - v0 = {_format_number(speed_delta)} m/s.",
                f"Compute the elastic term: (EA/L1)*(v1 - v0) = {_format_number(elastic_term)} N/s.",
                f"Compute the convective term: (T0*v0 - T1*v1)/L1 = {_format_number(convective_term)} N/s.",
                f"Add both terms to obtain dT1/dt = {_format_number(d_t1)} N/s.",
            ],
        ),
    ]

    return {
        "calculation_summary": (
            "Simulation calculations recompute the reported tracking, effort, and one governing-equation derivative "
            "from the generated rows."
        ),
        "calculations": calculations,
    }


def sysid_calculation_payload(
    metrics: Mapping[str, float],
    error_table: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return one worked example for each SysID parameter plus MARE_theta."""

    calculations: list[dict[str, object]] = []
    rel_errors: list[float] = []
    for row in error_table:
        name = str(row["parameter"])
        estimate = float(row["estimate"])
        truth = float(row["truth"])
        relative_error = _safe_relative(estimate, truth)
        rel_errors.append(relative_error)
        calculations.append(
            _calculation(
                title=f"{name} Relative Error",
                parameter=name,
                formula="(estimate - truth) / truth",
                values={
                    "estimate": _clean_number(estimate),
                    "truth": _clean_number(truth),
                    "absolute_error": _clean_number(estimate - truth),
                    "relative_error": _clean_number(relative_error),
                },
                substitution=f"({_format_number(estimate)} - {_format_number(truth)}) / {_format_number(truth)}",
                result=f"{name} relative_error = {_format_number(relative_error)}",
                summary="This parameter contribution enters MARE_theta through its absolute value.",
                steps=[
                    f"Read the SysID estimate for {name}: {_format_number(estimate)}.",
                    f"Read the true/reference value for {name}: {_format_number(truth)}.",
                    f"Compute absolute error: estimate - truth = {_format_number(estimate - truth)}.",
                    f"Normalize by truth: {_format_number(estimate - truth)} / {_format_number(truth)} = {_format_number(relative_error)}.",
                    "The absolute value of this relative error is later included in MARE_theta.",
                ],
            )
        )

    if rel_errors:
        mare_theta = sum(abs(error) for error in rel_errors) / len(rel_errors)
        calculations.append(
            _calculation(
                title="SysID Parameter MARE",
                parameter="MARE_theta",
                formula="mean(abs(relative_error_i))",
                values={
                    "parameters": float(len(rel_errors)),
                    "absolute_relative_error_sum": _clean_number(sum(abs(error) for error in rel_errors)),
                },
                substitution=(
                    f"{_format_number(sum(abs(error) for error in rel_errors))} / {len(rel_errors)}"
                ),
                result=f"MARE_theta = {_format_number(mare_theta)}",
                summary="A lower MARE_theta means the identified parameters are closer to the true model parameters.",
                steps=[
                    f"Collect the {len(rel_errors)} relative errors from kt_UW, kt_Nip, kt_RW, kf_UW, kf_Nip, kf_RW, and EA.",
                    "Take the absolute value of each relative error so signs do not cancel.",
                    f"Add absolute relative errors: {_format_number(sum(abs(error) for error in rel_errors))}.",
                    f"Divide by parameter count: {len(rel_errors)}.",
                    f"Obtain MARE_theta = {_format_number(mare_theta)}.",
                ],
            )
        )

    return {
        "calculation_summary": (
            "SysID calculations show one relative-error example for every estimated parameter, then combine them "
            "into MARE_theta."
        ),
        "calculations": calculations,
    }


def _study_metrics(payload: Mapping[str, object]) -> Mapping[str, object]:
    metrics = payload.get("metrics", {})
    return metrics if isinstance(metrics, Mapping) else {}


def _inner_rows(study_metrics: Mapping[str, object]) -> list[Mapping[str, object]]:
    rows = study_metrics.get("metrics", [])
    if not isinstance(rows, Sequence):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def logging_rate_calculation_payload(payload: Mapping[str, object]) -> dict[str, object]:
    study_metrics = _study_metrics(payload)
    rows = _inner_rows(study_metrics)
    # The selected logging period is the DUAL-CHANNEL optimum. Tension-only
    # noise has no interior optimum, so its minimum is always the finest period
    # and would misreport the recommendation as "log as fast as you can".
    noisy_rows = [row for row in rows if row.get("case") == "dual_channel"]
    best = min(noisy_rows, key=lambda row: float(row["MARE_theta"])) if noisy_rows else None
    power_law_rows = [
        row for row in noisy_rows if row.get("reference_power_law_MARE_theta") is not None
    ]
    best_power_law = (
        min(power_law_rows, key=lambda row: float(row["reference_power_law_MARE_theta"]))
        if power_law_rows
        else None
    )
    calculations: list[dict[str, object]] = []
    if best:
        best_tlog = float(best["Tlog_ms"])
        best_mare = float(best["MARE_theta"])
        best_mare_percent = float(best.get("MARE_theta_percent", best.get("MARE_theta_percent", best_mare * 100.0)))
        best_tmin = float(best.get("tmin_ms", study_metrics.get("tmin_ms", 50.0)))
        tau_ratio = float(best.get("tau_ratio", best_tlog / best_tmin))
        tau_min_over_tlog = float(best.get("tau_min_over_tlog", best_tmin / best_tlog))
        calculations.append(
            _calculation(
                title="Best Dual-Channel Logging Rate",
                parameter="best_noisy_Tlog_ms",
                formula="argmin_Tlog MARE_theta(Tlog, dual_channel)",
                values={
                    "tested_noisy_Tlog_ms": ", ".join(_format_number(float(row["Tlog_ms"])) for row in noisy_rows),
                    "best_noisy_Tlog_ms": _clean_number(best_tlog),
                    "best_noisy_MARE_theta": _clean_number(best_mare),
                    "best_noisy_MARE_theta_percent": _clean_number(best_mare_percent),
                },
                substitution=(
                    f"minimum noisy MARE_theta is {_format_number(best_mare)} "
                    f"({ _format_number(best_mare_percent) }%) at Tlog={_format_number(best_tlog)} ms"
                ),
                result=f"best_noisy_Tlog_ms = {_format_number(best_tlog)} ms",
                summary="The selected logging rate is the dual-channel case with the lowest parameter-estimation error; the interior optimum is produced by the velocity channel.",
                steps=[
                    "Filter the sweep table to rows where case = dual_channel.",
                    "For each noisy logging rate, run SysID and record MARE_theta.",
                    f"Compare tested noisy Tlog values: {', '.join(_format_number(float(row['Tlog_ms'])) for row in noisy_rows)} ms.",
                    f"Select the row with minimum MARE_theta: {_format_number(best_mare)}.",
                    f"Convert to percent: MARE_theta_percent = 100*MARE_theta = {_format_number(best_mare_percent)}%.",
                    f"Report its logging period: Tlog = {_format_number(best_tlog)} ms.",
                ],
            )
        )
        if best_power_law:
            power_tlog = float(best_power_law["Tlog_ms"])
            power_tmin = float(best_power_law.get("tmin_ms", study_metrics.get("tmin_ms", 50.0)))
            power_tau_ratio = float(best_power_law.get("tau_ratio", power_tlog / power_tmin))
            power_mare_percent = float(best_power_law["reference_power_law_MARE_theta_percent"])
            power_mare = float(best_power_law["reference_power_law_MARE_theta"])
            power_a_raw = best_power_law.get("reference_power_law_a")
            power_alpha_raw = best_power_law.get("reference_power_law_alpha")
            if power_a_raw is not None and power_alpha_raw is not None:
                power_a = float(power_a_raw)
                power_alpha = float(power_alpha_raw)
                calculations.append(
                    _calculation(
                        title="tmin-Dependent MARE",
                        parameter="reference_power_law_MARE_theta",
                        formula="MARE_theta_percent = a * (Tlog/tmin)^alpha; MARE_theta = MARE_theta_percent/100",
                        values={
                            "Tlog_ms": _clean_number(power_tlog),
                            "tmin_ms": _clean_number(power_tmin),
                            "tau_ratio": _clean_number(power_tau_ratio),
                            "a": _clean_number(power_a),
                            "alpha": _clean_number(power_alpha),
                            "reference_power_law_MARE_theta_percent": _clean_number(power_mare_percent),
                            "reference_power_law_MARE_theta": _clean_number(power_mare),
                        },
                        substitution=(
                            f"{_format_number(power_a)}*({_format_number(power_tlog)}/"
                            f"{_format_number(power_tmin)})^{_format_number(power_alpha)}"
                        ),
                        result=f"reference_power_law_MARE_theta = {_format_number(power_mare)}",
                        summary="This paper-reference MARE estimate is tied directly to Tlog/tmin, so changing tmin changes the calculated value.",
                        steps=[
                            f"Use the paper-reference noisy power-law fit coefficient a = {_format_number(power_a)}.",
                            f"Use alpha = {_format_number(power_alpha)} from the same fit.",
                            f"Compute Tlog/tmin = {_format_number(power_tlog)} / {_format_number(power_tmin)} = {_format_number(power_tau_ratio)}.",
                            f"Evaluate MARE_theta_percent = a*(Tlog/tmin)^alpha = {_format_number(power_mare_percent)}%.",
                            f"Convert percent to MARE_theta by dividing by 100: {_format_number(power_mare)}.",
                        ],
                    )
                )
            else:
                calculations.append(
                    _calculation(
                        title="Numerical Paper SN Reference",
                        parameter="reference_power_law_MARE_theta",
                        formula="MARE_theta = MARE_theta_percent/100",
                        values={
                            "Tlog_ms": _clean_number(power_tlog),
                            "tmin_ms": _clean_number(power_tmin),
                            "tau_ratio": _clean_number(power_tau_ratio),
                            "reference_MARE_theta_percent": _clean_number(power_mare_percent),
                            "reference_MARE_theta": _clean_number(power_mare),
                        },
                        substitution=f"{_format_number(power_mare_percent)} / 100",
                        result=f"reference_MARE_theta = {_format_number(power_mare)}",
                        summary="The noisy paper comparison uses the numerical U-shaped SN curve from the paper text.",
                        steps=[
                            "Use the paper SN values reported for the logging-rate numerical comparison.",
                            f"At Tlog={_format_number(power_tlog)} ms, read/interpolate MARE_theta_percent = {_format_number(power_mare_percent)}%.",
                            f"Convert percent to MARE_theta by dividing by 100: {_format_number(power_mare)}.",
                        ],
                    )
                )
        calculations.append(
            _calculation(
                title="Logging Adequacy Ratio",
                parameter="tau_min_over_tlog",
                formula="tau_ratio = Tlog/tmin; tau_min_over_tlog = tmin/Tlog",
                values={
                    "Tlog_ms": _clean_number(best_tlog),
                    "tmin_ms": _clean_number(best_tmin),
                    "tau_ratio": _clean_number(tau_ratio),
                    "tau_min_over_tlog": _clean_number(tau_min_over_tlog),
                    "nf_guideline_tau_min_over_tlog": 5.0,
                },
                substitution=(
                    f"tau_ratio={_format_number(best_tlog)}/{_format_number(best_tmin)}="
                    f"{_format_number(tau_ratio)}; tau_min/Tlog={_format_number(best_tmin)}/"
                    f"{_format_number(best_tlog)}={_format_number(tau_min_over_tlog)}"
                ),
                result=f"tau_min_over_tlog = {_format_number(tau_min_over_tlog)}",
                summary="This is the ratio used by the paper's noise-free logging guideline.",
                steps=[
                    f"Use tmin = tau_min = {_format_number(best_tmin)} ms.",
                    f"Use the selected noisy Tlog = {_format_number(best_tlog)} ms.",
                    f"Compute Tlog/tmin = {_format_number(tau_ratio)}.",
                    f"Compute tau_min/Tlog = {_format_number(tau_min_over_tlog)}.",
                    "For the noise-free case, the paper guideline is tau_min/Tlog >= 5.",
                ],
            )
        )
        calculations.append(
            _calculation(
                title="5-20 ms Window Check",
                parameter="supports_noisy_optimum_in_5_20ms_window",
                formula="best_noisy_Tlog_ms in {5, 10, 20}",
                values={
                    "best_noisy_Tlog_ms": _clean_number(best_tlog),
                    "paper_window_ms": "5, 10, 20",
                    "paper_pooled_optimum_ms": 5,
                },
                substitution=f"{_format_number(best_tlog)} in {{5, 10, 20}}",
                result=(
                    "supports_noisy_optimum_in_5_20ms_window = "
                    f"{str(best_tlog in (5.0, 10.0, 20.0)).lower()}"
                ),
                summary=(
                    "This validates whether the simulated noisy optimum falls in the "
                    "noise-aware logging window. Under dual-channel noise the pooled "
                    "interior optimum sits at 5 ms; the position is not universal, so the "
                    "paper gives a 5-20 ms window set by the excitation and the plant's "
                    "velocity-noise dose. Tension-only noise has no interior optimum at "
                    "all - its optimum is the finest 1 ms setting."
                ),
                steps=[
                    f"Use the previously selected noisy optimum Tlog = {_format_number(best_tlog)} ms.",
                    "Compare it with the paper-supported window {5 ms, 10 ms, 20 ms}.",
                    f"Return true only if {_format_number(best_tlog)} equals 5, 10 or 20.",
                ],
            )
        )
        paper_median_raw = best.get("paper_median_MARE_theta_percent")
        if paper_median_raw is not None:
            paper_median = float(paper_median_raw)
            delta = best_mare_percent - paper_median
            calculations.append(
                _calculation(
                    title="Paper Median Delta",
                    parameter="paper_delta_median_MARE_theta_percent",
                    formula="dashboard_MARE_theta_percent - paper_median_MARE_theta_percent",
                    values={
                        "dashboard_MARE_theta_percent": _clean_number(best_mare_percent),
                        "paper_median_MARE_theta_percent": _clean_number(paper_median),
                        "delta_MARE_theta_percent": _clean_number(delta),
                    },
                    substitution=f"{_format_number(best_mare_percent)} - {_format_number(paper_median)}",
                    result=f"paper_delta_median_MARE_theta_percent = {_format_number(delta)}",
                    summary="This compares the current dashboard run with the Fig. 02 CSV median at the same Tlog.",
                    steps=[
                        f"Read dashboard MARE_theta_percent at Tlog={_format_number(best_tlog)} ms: {_format_number(best_mare_percent)}%.",
                        f"Read the paper-reference median MARE_theta_percent for the same Tlog: {_format_number(paper_median)}%.",
                        f"Subtract paper median from dashboard value to get {_format_number(delta)}%.",
                    ],
                )
            )
    return {
        "calculation_summary": (
            "Logging-rate calculations solve the noisy optimum, compute Tlog/tmin adequacy ratios, "
            "compute the paper comparison MARE, and compare matching rows with the Fig. 02 paper reference."
        ),
        "calculations": calculations,
    }


def excitation_calculation_payload(payload: Mapping[str, object]) -> dict[str, object]:
    study_metrics = _study_metrics(payload)
    rows_source = study_metrics.get("comparison_rows", _inner_rows(study_metrics))
    rows = [row for row in rows_source if isinstance(row, Mapping)] if isinstance(rows_source, Sequence) else []
    best = min(rows, key=lambda row: float(row["dashboard_SN_percent"])) if rows else None
    calculations: list[dict[str, object]] = []
    if best:
        excitation = str(best.get("excitation", best.get("strategy")))
        mare_percent = float(best["dashboard_SN_percent"])
        paper_percent = best.get("paper_SN_percent")
        difference_percent = best.get("difference_SN_percent")
        multi_channel = excitation in {"ET3", "ET6", "ET3M", "E_Toggle"}
        calculations.append(
            _calculation(
                title="Best Noisy Excitation",
                parameter="best_noisy_excitation",
                formula="argmin_excitation dashboard_SN_percent(excitation)",
                values={
                    "tested_noisy_excitations": ", ".join(str(row.get("strategy", row.get("excitation"))) for row in rows),
                    "best_noisy_excitation": excitation,
                    "best_noisy_MARE_theta_percent": _clean_number(mare_percent),
                },
                substitution=f"minimum noisy MARE_theta percent is {_format_number(mare_percent)} for {excitation}",
                result=f"best_noisy_excitation = {excitation}",
                summary="The excitation with the smallest noisy MARE_theta gives the strongest validation signal for SysID.",
                steps=[
                    "Use the backend comparison rows, where each row contains the median SN MARE_theta percent.",
                    "For each excitation profile, run backend simulation and SysID for the selected plant scope.",
                    f"Compare profiles: {', '.join(str(row.get('strategy', row.get('excitation'))) for row in rows)}.",
                    f"Select the minimum SN MARE_theta percent row: {_format_number(mare_percent)}% for {excitation}.",
                ],
            )
        )
        if paper_percent is not None and difference_percent is not None:
            calculations.append(
                _calculation(
                    title="Excitation Paper Difference",
                    parameter="difference_SN_percent",
                    formula="dashboard_SN_percent - paper_SN_percent",
                    values={
                        "excitation": excitation,
                        "dashboard_SN_percent": _clean_number(mare_percent),
                        "paper_SN_percent": _clean_number(float(paper_percent)),
                    },
                    substitution=f"{_format_number(mare_percent)} - {_format_number(float(paper_percent))}",
                    result=f"difference_SN_percent = {_format_number(float(difference_percent))}",
                    summary="This compares the backend dashboard result with the paper/reference value without changing the dashboard value.",
                    steps=[
                        f"Read backend dashboard SN MARE_theta percent for {excitation}: {_format_number(mare_percent)}%.",
                        f"Read reference SN value from data/reference_results: {_format_number(float(paper_percent))}%.",
                        f"Subtract paper from dashboard: {_format_number(float(difference_percent))} percentage points.",
                    ],
                )
            )
        calculations.append(
            _calculation(
                title="Multi-Channel Excitation Check",
                parameter="supports_multi_channel_or_toggle_under_noise",
                formula="best_noisy_excitation in {ET3, ET6, ET3M, E_Toggle}",
                values={
                    "best_noisy_excitation": excitation,
                    "accepted_profiles": "ET3, ET6, ET3M, E_Toggle",
                },
                substitution=f"{excitation} in {{ET3, ET6, ET3M, E_Toggle}}",
                result=f"supports_multi_channel_or_toggle_under_noise = {str(multi_channel).lower()}",
                summary="This checks whether a multi-channel or toggle excitation wins under noisy measurement conditions.",
                steps=[
                    f"Use best noisy excitation = {excitation}.",
                    "Compare with accepted multi-channel/toggle profiles: ET3, ET6, ET3M, E_Toggle.",
                    f"Return {str(multi_channel).lower()} for the multi-channel/toggle check.",
                ],
            )
        )
    return {
        "calculation_summary": "Excitation calculations identify the lowest-MARE noisy excitation and classify its type.",
        "calculations": calculations,
    }


def drift_calculation_payload(payload: Mapping[str, object]) -> dict[str, object]:
    study_metrics = _study_metrics(payload)
    rows_source = payload.get("comparison_rows")
    if not isinstance(rows_source, Sequence) or isinstance(rows_source, (str, bytes)):
        rows_source = study_metrics.get("comparison_rows", _inner_rows(study_metrics))
    rows = list(rows_source)
    # Only legs the paper actually prints an NF median for can carry a paper
    # difference. The EA legs are published as a band, never per leg, so a row
    # with `paper_NF_percent = None` must not be picked as the comparison case.
    comparable_rows = [row for row in rows if row.get("paper_NF_percent") is not None]
    dominant = (
        max(comparable_rows, key=lambda row: float(row["dashboard_SN_percent"]))
        if comparable_rows
        else None
    )
    calculations: list[dict[str, object]] = []
    if dominant:
        drift_case = str(dominant["drift_case"])
        dashboard_sn = float(dominant["dashboard_SN_percent"])
        dashboard_nf = float(dominant["dashboard_NF_percent"])
        paper_nf = float(dominant["paper_NF_percent"])
        nf_difference = dashboard_nf - paper_nf
        paper_sn_value = dominant.get("paper_SN_percent")
        if paper_sn_value is not None:
            paper_sn = float(paper_sn_value)
            sn_difference = dashboard_sn - paper_sn
            calculations.append(
                _calculation(
                    title="Drift Paper Difference",
                    parameter="difference_SN_percent",
                    formula="dashboard_SN_percent - paper_SN_percent",
                    values={
                        "drift_case": drift_case,
                        "dashboard_SN_percent": _clean_number(dashboard_sn),
                        "paper_SN_percent": _clean_number(paper_sn),
                    },
                    substitution=f"{_format_number(dashboard_sn)} - {_format_number(paper_sn)}",
                    result=f"{drift_case} SN difference = {_format_number(sn_difference)} percentage points",
                    summary="The Drift tab compares dashboard median SysID error against the extracted paper reference where a paper SN value is available.",
                    steps=[
                        f"Use the largest dashboard SN drift case with a paper SN reference: {drift_case}.",
                        f"Read dashboard SN median SysID error = {_format_number(dashboard_sn)}%.",
                        f"Read paper SN reference = {_format_number(paper_sn)}%.",
                        f"Subtract paper from dashboard to get {_format_number(sn_difference)} percentage points.",
                    ],
                )
            )
        calculations.append(
            _calculation(
                title="Noise-Free Difference",
                parameter="difference_NF_percent",
                formula="dashboard_NF_percent - paper_NF_percent",
                values={
                    "drift_case": drift_case,
                    "dashboard_NF_percent": _clean_number(dashboard_nf),
                    "paper_NF_percent": _clean_number(paper_nf),
                },
                substitution=f"{_format_number(dashboard_nf)} - {_format_number(paper_nf)}",
                result=f"{drift_case} NF difference = {_format_number(nf_difference)} percentage points",
                summary="This repeats the paper comparison for the noise-free curve.",
                steps=[
                    f"Use the same drift case: {drift_case}.",
                    f"Read dashboard NF median SysID error = {_format_number(dashboard_nf)}%.",
                    f"Read paper NF reference = {_format_number(paper_nf)}%.",
                    f"Subtract paper from dashboard to get {_format_number(nf_difference)} percentage points.",
                ],
            )
        )
    return {
        "calculation_summary": "Drift calculations compare dashboard median SysID error with paper reference values for NF and SN.",
        "calculations": calculations,
    }

