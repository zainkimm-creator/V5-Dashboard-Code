"""Tests for the BO-configuration sensitivity block of the retuning payload.

Section 4.2's headline ordering (twin transfer vs. cold-start BO) turns out to
be a function of the baseline's *unprinted* skopt sampling prior: with our
log-uniform prior CS-BO(30) converges to the per-cell achievable bound and the
twin can never win a strict-inequality pair, while a linear-uniform prior on
absolute gains leaves it short and the twin wins ~42% of cells. The dashboard
carries that sensitivity as data so the panel can show the band rather than
asserting one end of it.

Like the other Section-4 tests these skip cleanly where the offline artefacts
are absent rather than failing the suite.
"""

from __future__ import annotations

import warnings

import pytest

warnings.filterwarnings("ignore")

from fastapi.testclient import TestClient  # noqa: E402

from backend.api.main import app  # noqa: E402
from backend.validation.retuning_results import (  # noqa: E402
    BO_CONFIG_STUDY,
    CANONICAL_CAMPAIGN,
    REPORTS_DIR,
    bo_config_sensitivity,
)

client = TestClient(app)

campaign_present = (REPORTS_DIR / CANONICAL_CAMPAIGN / "cells").exists()
study_present = BO_CONFIG_STUDY.exists()

needs_artifacts = pytest.mark.skipif(
    not (campaign_present and study_present),
    reason="offline campaign or BO-configuration study not run here",
)


@pytest.fixture(scope="module")
def block() -> dict:
    return bo_config_sensitivity()


@needs_artifacts
class TestBoConfigSensitivity:
    def test_block_is_available_and_names_its_source(self, block) -> None:
        assert block["available"] is True
        assert "bo_config_study" in block["source"]

    def test_carries_every_studied_configuration_plus_ours_and_the_paper(self, block) -> None:
        keys = [v["key"] for v in block["variants"]]
        assert keys == ["V-base", "V-wide", "V-linabs", "V-cluster", "paper"]

    def test_exactly_one_variant_is_flagged_as_the_dashboard_default(self, block) -> None:
        defaults = [v for v in block["variants"] if v.get("is_default")]
        assert [v["key"] for v in defaults] == ["V-base"]

    def test_linear_uniform_prior_is_the_lever_that_moves_the_win_rate(self, block) -> None:
        by_key = {v["key"]: v for v in block["variants"]}
        # Reproduced from reports/bo_config_study/results.json: 25 of 60 cells.
        assert by_key["V-linabs"]["hgs_win_percent"] == pytest.approx(41.7, abs=0.5)
        # Box width and warm-start seeding do not move it at all.
        assert by_key["V-wide"]["hgs_win_percent"] == pytest.approx(0.0, abs=0.1)
        assert by_key["V-cluster"]["hgs_win_percent"] == pytest.approx(0.0, abs=0.1)
        assert by_key["V-base"]["hgs_win_percent"] == pytest.approx(0.0, abs=0.1)

    def test_cluster_seeding_reproduces_the_papers_pinned_warm_start(self, block) -> None:
        by_key = {v["key"]: v for v in block["variants"]}
        # The paper's WS-BO sits flat at 10.7 through five evaluations; a single
        # seed point escapes by eval 3, only a cluster stays pinned high.
        assert by_key["V-cluster"]["median_at_5"] > by_key["V-base"]["median_at_5"] * 3
        assert by_key["V-cluster"]["median_at_30"] == pytest.approx(
            by_key["V-base"]["median_at_30"], abs=0.02)

    def test_band_spans_our_result_the_alternate_prior_and_the_paper(self, block) -> None:
        band = block["band"]
        assert band["min_percent"] == pytest.approx(0.0, abs=0.1)
        assert band["max_percent"] == pytest.approx(41.7, abs=0.5)
        assert band["paper_percent"] == pytest.approx(58.0, abs=0.1)

    def test_paper_row_is_reference_only_and_carries_no_local_pair_count(self, block) -> None:
        paper = [v for v in block["variants"] if v["key"] == "paper"][0]
        assert paper["reference_only"] is True
        assert paper["n_pairs"] is None
        assert paper["hgs_win_percent"] == pytest.approx(58.0, abs=0.1)

    def test_every_local_variant_reports_how_many_runs_back_it(self, block) -> None:
        for v in block["variants"]:
            if v["key"] == "paper":
                continue
            assert v["n_runs"] > 0
            assert v["n_pairs"] == 60

    def test_route_exposes_the_block(self) -> None:
        payload = client.post("/validate/retuning").json()
        assert "bo_config_sensitivity" in payload
        assert payload["bo_config_sensitivity"]["available"] is True


class TestBoConfigSensitivityDegradesCleanly:
    def test_missing_study_reports_unavailable_rather_than_raising(self, tmp_path) -> None:
        block = bo_config_sensitivity(study_path=tmp_path / "absent.json")
        assert block["available"] is False
        assert block["reason"]
        assert block["variants"] == []
