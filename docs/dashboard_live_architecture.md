# Dashboard-first live calculation architecture

## Status and source of truth

This document defines how the dashboard calculates its own results and how it
compares them against the paper.

The implementation under `V2_Drift_Review41_Recalc_20260723` is the current
code source of truth. The paper and historical dashboard packages are optional
comparison sources only. They must never supply a calculated dashboard value.
The old `section4-hpc-bo` skill is not an implementation authority for this
contract.

## Current architecture

| Layer | Current dashboard-first implementation |
| --- | --- |
| Frontend | React 19 and Vite; one page per validation section, each auto-loading its all-plant study and rendering the paper comparison |
| API | FastAPI validation routes in `backend/api/main.py` |
| Simulation | `backend/models/simulation.py` plus the current reviewed controller/model modules; every study freshly simulates each plant |
| SysID | `backend/sysid/estimator.py`; the v5 canonical estimator is the operating-point-weighted PEM (Eq. 8) |
| Validation studies | All-plant paper-comparison studies in `backend/validation/studies.py`, `noise_aware_logging_lpf.py` and `closed_loop_damping.py` |

## Measurement conditions

`noise.condition` is the discriminator every module acts on. It takes three
values, matching how the paper scopes every result:

| Value | Meaning |
|---|---|
| `noise_free` | The commissioning reference. No sensor noise on either channel. |
| `tension_only` | Noise on the load-cell channel alone. |
| `dual_channel` | Noise on tension **and** velocity, which is what a real line has. |

`enabled` is retained so existing clients and stored sessions keep working, and
the two are reconciled in both directions: clearing `enabled` forces
`noise_free`, and selecting `noise_free` clears `enabled`.

The distinction is not cosmetic. The interior logging optimum is produced by the
**velocity** channel; tension noise alone never creates one, and its optimum is
always the finest logging period. Any result that does not carry its condition
tag is ambiguous, so every module stamps `measurement_condition`, `pct_T`,
`pct_v` and both LPF cutoffs onto its summary.

`velocity_level_full_scale_percent` and `velocity_lpf_cutoff_hz` default to
their tension counterparts, because the paper quotes the common dual-channel
dose as a single number (0.3 %/0.3 %, LPF 50/50 Hz).

Noise model:

```text
sigma_T = pct_T * T_max
sigma_v = pct_v * v_max,  v_max = v0 / 0.30
```

`sigma_v` is a surface-speed error injected as the *same* speed error on every
roller, so a roller of radius `R` sees `sigma_v / R` rad/s. Noise is added to the
sensor channels only, never in the ODE right-hand side.

Note that `T_max` defines the noise model only. The estimator's weight matrix
uses the controller set-point `T_ref`; the two must not be interchanged.

## Estimator policy

`estimation.estimator_id` accepts:

| Value | Meaning |
|---|---|
| `module_default` | Resolves to `paper_eq8_weighted_pem_trf`. |
| `paper_eq8_weighted_pem_trf` | **The canonical v5 estimator.** Operating-point-weighted PEM on the six-channel logged state. |
| `paper_eq7_one_step_pem_trf` | The v4.1 unweighted tension-only PEM, retained for comparison views only. |
| `legacy_finite_difference_lsq` | The original closed-form finite-difference estimator. |

Accuracy numbers from the weighted and unweighted estimators are **not**
interchangeable. A comparison view may show both, but it must label which is
which.

The Fisher information is `F = sum_k J_k^T W^2 J_k`. On the weighted-residual
scale the absolute value of `kappa(F)` depends on the operating-point
normalization, so only the ordering across plants or excitations is
interpretable. The estimator returns `log10_kappa_fisher` for ranking; do not
present it as an absolute figure.

## Paper comparison in the validation views

The all-plant validation studies below the live panels compare against the paper.
Two rules govern how a published value reaches the screen.

**Every published cell must be reachable.** v5 publishes a different Tlog span
per measurement condition: a complete 2-100 ms noise-free series (supplement
Table S7, ET1), a complete 1-100 ms tension-only series (supplement Fig. S6(b)
at 50 Hz, E_Toggle), and three dual-channel points (Fig. 2(a), E_Toggle at 5, 20
and 50 ms). The Logging Rate study therefore sweeps all three conditions and
compares each against its own series. Collapsing the noisy conditions onto one
"sensor noise" slot made the other two series unreachable, which is why the
comparison table used to show `n/a` for logging periods the paper does print.

**A cell with no published number says why.** A blank reads as "the dashboard
lost the value". The two honest reasons are stamped instead:

| Status | Shown as | Meaning |
|---|---|---|
| `published` | the number | The paper prints this cell. |
| `off_scale_below_0p01_percent` | `off-scale (<0.01%)` | Noise-free at Tlog = 1 ms is an essentially exact fit that falls off a logarithmic axis. Fig. 2(a) starts its noise-free curve at 2 ms for this reason. |
| `not_published_at_this_Tlog` | `not published` | The paper prints no value for this cell. |

The Drift study has the same problem in a different shape: Section 3.3 publishes
the EA family as a **band** across its five legs, not per leg, and the friction
family as a noise-free **pair** only. Per-leg paper cells are null by design, so
the band lives on `paper_family_reference` at family level and reaches the UI
three ways: as `band 24.4-25.2%` in each EA row's paper column, as a shaded band
on the EA figure, and as a Paper Reference panel beside the comparison table.
Synthesizing a per-leg value out of a band would misreport the paper and is not
done.

## Figure map

The v5 figure numbers do not carry over from v4.1. The Logging Rate view names
its figures explicitly:

| Tab | Figure | Contents |
|---|---|---|
| Three conditions | v5 Fig. 2(a) | `epsilon_theta` vs Tlog under noise-free, tension-only and dual-channel noise. Dashboard solid, paper dashed, one colour per condition. |
| Speed split | v5 Fig. 2(b) | The dual-channel case decomposed by line speed, using the paper's own slow/fast plant lists, with the pooled median drawn dotted for contrast. |
| Power law | v5 Fig. S2 | The `tau_min`-normalized **noise-free** power law. |

The power law is fitted on the noise-free branch only. The sensor-noise branch is
deliberately not overlaid: under tension-only sensor noise the error does not
collapse onto it (fitted exponent +0.41, R^2 ~ 0.11), so an SN line would
contradict its own trend.
