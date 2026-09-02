"""Regression tests for the Tier 1 Section 4.2 check.

These lock the reporting layer: seed pooling, the percentile convention, the
method-name mapping, and the agreement between the paper, the figure package
and `retuning_reference.json`.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.validation.retuning_tier1 import (
    DEFAULT_FIGURE_PACKAGE,
    PAPER_METHODS,
    PAPER_N,
    PAPER_TABLE_S9,
    describe,
    load_per_run_costs,
    percentile_linear,
    run_tier1,
)

from backend.validation.paper_reference import comparison_data_available  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (DEFAULT_FIGURE_PACKAGE.exists() and comparison_data_available()),
    reason="v5 figure package not present beside the dashboard",
)


@pytest.fixture(scope="module")
def result():
    return run_tier1()


# --------------------------------------------------------------------------- #
# the percentile convention is the thing most likely to drift
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q", [5.0, 25.0, 50.0, 75.0, 95.0])
@pytest.mark.parametrize("size", [1, 2, 7, 60, 180, 541])
def test_percentile_matches_numpy(q: float, size: int) -> None:
    rng = np.random.default_rng(size * 1000 + int(q))
    values = rng.lognormal(mean=-0.5, sigma=1.2, size=size).tolist()
    assert percentile_linear(values, q) == pytest.approx(
        float(np.percentile(values, q)), abs=1e-12
    )


def test_percentile_rejects_empty() -> None:
    with pytest.raises(ValueError):
        percentile_linear([], 50.0)


def test_describe_on_known_values() -> None:
    stats = describe([1.0, 2.0, 3.0, 4.0])
    assert stats["n"] == 4
    assert stats["mean"] == pytest.approx(2.5)
    assert stats["median"] == pytest.approx(2.5)
    assert stats["min"] == 1.0
    assert stats["max"] == 4.0


# --------------------------------------------------------------------------- #
# seed structure
# --------------------------------------------------------------------------- #
def test_pooled_run_counts() -> None:
    per_run = load_per_run_costs(DEFAULT_FIGURE_PACKAGE)
    assert sum(len(v) for v in per_run.values()) == 540
    for name in PAPER_METHODS:
        assert len(per_run[name]) == PAPER_N[name], name


def test_stochastic_methods_carry_three_seeds() -> None:
    per_run = load_per_run_costs(DEFAULT_FIGURE_PACKAGE)
    # 60 plant x drift combinations; CS/WS-BO replicate over 3 BO seeds.
    assert len(per_run["CS-BO(30)"]) == 3 * 60
    assert len(per_run["HGS-only"]) == 1 * 60


# --------------------------------------------------------------------------- #
# paper agreement
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", PAPER_METHODS)
def test_matches_paper_table_s9(result, method: str) -> None:
    stats = result.method_stats[method]
    for stat, expected in PAPER_TABLE_S9[method].items():
        assert round(stats[stat], 3) == pytest.approx(expected, abs=5e-4), (
            f"{method}/{stat}"
        )


def test_hgs_family_shares_one_median(result) -> None:
    medians = [result.method_stats[m]["median"]
               for m in ("HGS-only", "HGS+BO(5)", "HGS+BO(10)")]
    # The three arms agree to ~1e-16; they differ only in float accumulation
    # order, so this is an exact-agreement test at double precision, not a
    # loose tolerance.
    assert max(medians) - min(medians) < 1e-12, (
        "few-shot BO must add no median gain under v5"
    )


def test_hgs_only_no_worse_than_cold_start(result) -> None:
    assert (result.method_stats["HGS-only"]["median"]
            <= result.method_stats["CS-BO(30)"]["median"])


def test_tails_are_method_independent(result) -> None:
    p95 = [result.method_stats[m]["P95"] for m in PAPER_METHODS]
    # Supplement S8: the worst case is set by plant heterogeneity, so the
    # upper tail must barely move across methods.
    assert max(p95) - min(p95) < 0.06


# --------------------------------------------------------------------------- #
# the whole gate
# --------------------------------------------------------------------------- #
def test_tier1_passes(result) -> None:
    failures = [c for c in result.checks if c.status == "FAIL"]
    assert not failures, [f"{c.group}/{c.name}" for c in failures]
    assert result.status == "PASS"


def test_unverifiable_items_are_declared_not_passed(result) -> None:
    unver = [c for c in result.checks if c.status == "UNVERIFIABLE"]
    # The paired win rates and sim-to-real gaps need plant/drift columns the
    # distributed package does not ship; they must never count as passes.
    assert len(unver) >= 6
    assert all(c.expected is None and c.actual is None for c in unver)
