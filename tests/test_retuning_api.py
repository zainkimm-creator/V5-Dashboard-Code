"""API tests for the Section 4 retuning routes.

The results route reads precomputed campaign checkpoints (the offline pattern
the Section-4 guide prescribes), so these tests skip cleanly on a machine that
has not run the campaign rather than failing the suite.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")

from fastapi.testclient import TestClient  # noqa: E402

from backend.api.main import app  # noqa: E402
from backend.validation.retuning_results import (  # noqa: E402
    CANONICAL_CAMPAIGN,
    REPORTS_DIR,
)

client = TestClient(app)

campaign_present = (REPORTS_DIR / CANONICAL_CAMPAIGN / "cells").exists()


@pytest.mark.skipif(not campaign_present, reason="offline campaign not run here")
class TestRetuningResults:
    def test_route_returns_full_payload(self) -> None:
        response = client.post("/validate/retuning")
        assert response.status_code == 200
        payload = response.json()
        for key in ("campaign", "provenance", "caveats", "protocols",
                    "convergence", "diagnostic_runs", "reports"):
            assert key in payload, key

    def test_methods_carry_paper_columns_as_comparison_only(self) -> None:
        payload = client.post("/validate/retuning").json()
        methods = payload["protocols"]["field_matched"]["methods"]
        names = [m["method"] for m in methods]
        assert names == ["CS-BO(30)", "WS-BO(30)", "HGS-only",
                         "HGS+BO(5)", "HGS+BO(10)"]
        for m in methods:
            assert m["real_evals"] in (0, 5, 10, 30)
            assert m["paper_median"] is not None
        assert payload["provenance"]["reference_data_usage"] == (
            "comparison_only_never_dashboard_calculation_input")

    def test_seed_structure_matches_paper(self) -> None:
        methods = {m["method"]: m for m in
                   client.post("/validate/retuning").json()
                   ["protocols"]["field_matched"]["methods"]}
        assert methods["CS-BO(30)"]["n"] == 180
        assert methods["HGS-only"]["n"] == 60

    def test_provenance_declares_reconstruction(self) -> None:
        provenance = client.post("/validate/retuning").json()["provenance"]
        assert "RECONSTRUCTED" in provenance["evaluation_model_status"]

    def test_convergence_has_paper_reference(self) -> None:
        conv = client.post("/validate/retuning").json()["convergence"]
        assert len(conv["ours_cs"]) == 30
        assert len(conv["paper_cs"]) == 30

    def test_diagnostics_are_labelled(self) -> None:
        for run in client.post("/validate/retuning").json()["diagnostic_runs"]:
            assert run["note"], run["campaign"]


def test_tier1_route() -> None:
    from backend.validation.paper_reference import comparison_data_available
    from backend.validation.retuning_tier1 import DEFAULT_FIGURE_PACKAGE

    response = client.post("/validate/retuning-tier1")
    # Tier 1 needs both the figure package and the published comparison set;
    # without either the route is expected to decline cleanly, not to fail.
    if not (DEFAULT_FIGURE_PACKAGE.exists() and comparison_data_available()):
        assert response.status_code == 422
        return
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "PASS"
    assert payload["failed"] == 0
    assert payload["passed"] >= 90
