"""Model inputs, and the optional published-result set used for comparison.

Two different things live behind this module, and the distinction matters:

* **Model inputs** (``data/model_inputs/``) define the simulated system - the
  physics constants, the ten-plant pool, the state-matrix structure and the
  cascade controller settings. They are required: nothing runs without them.
* **Comparison data** (``data/reference_results/``) is a set of previously
  published result values that the dashboard displays *beside* its own numbers
  so a reader can see the difference. It is **optional and not distributed
  here**. Comparison values never feed a calculation - every dashboard number
  is recomputed from the model.

With the comparison set absent - the normal case for this distribution - each
loader below returns an empty mapping, and every study drops its comparison
columns and reports its own values alone. Call ``comparison_data_available()``
to branch explicitly.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Required: the model definition.
INPUT_DIR = PROJECT_ROOT / "data" / "model_inputs"
MODEL_PARAMETERS_PATH = INPUT_DIR / "model_parameters.json"

# Optional: previously published results, for side-by-side display only.
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference_results"

EXCITATION_REFERENCE_PATH = REFERENCE_DIR / "excitation_reference.json"
DRIFT_REFERENCE_PATH = REFERENCE_DIR / "drift_reference.json"
LOGGING_POWER_LAW_REFERENCE_PATH = REFERENCE_DIR / "logging_power_law_reference.json"
LOGGING_RATE_REFERENCE_PATH = REFERENCE_DIR / "logging_rate_v5_reference.json"
NOISE_LPF_REFERENCE_PATH = REFERENCE_DIR / "noise_lpf_reference.json"
CLOSED_LOOP_DAMPING_REFERENCE_PATH = REFERENCE_DIR / "closed_loop_damping_reference.json"
RETUNING_REFERENCE_PATH = REFERENCE_DIR / "retuning_reference.json"
EXPERIMENT_LEDGER_PATH = REFERENCE_DIR / "experiment_ledger_v5.json"
FIGURES_PATH = REFERENCE_DIR / "figures_v5.json"


def comparison_data_available() -> bool:
    """True when an optional comparison set has been placed on disk."""

    return REFERENCE_DIR.is_dir() and any(REFERENCE_DIR.glob("*.json"))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_optional(path: Path) -> dict[str, Any]:
    """A comparison file, or an empty mapping when it is not distributed.

    Every caller treats an empty mapping as "no comparison column", so a
    missing file degrades the display rather than failing the request.
    """

    if not path.is_file():
        return {}
    return _load_json(path)


@lru_cache(maxsize=1)
def load_model_parameters() -> dict[str, Any]:
    """The model definition - required, and an error if it is missing."""

    if not MODEL_PARAMETERS_PATH.is_file():
        raise FileNotFoundError(
            f"Model definition not found at {MODEL_PARAMETERS_PATH}. "
            "data/model_inputs/ must contain model_parameters.json, "
            "ten_plant_parameters.csv and excitation_schedules.csv."
        )
    return _load_json(MODEL_PARAMETERS_PATH)


# Retained under the previous name so existing call sites keep working.
load_paper_reference = load_model_parameters


def paper_reference_path() -> str:
    return str(MODEL_PARAMETERS_PATH)


@lru_cache(maxsize=1)
def load_excitation_reference() -> dict[str, Any]:
    return _load_optional(EXCITATION_REFERENCE_PATH)


@lru_cache(maxsize=1)
def load_drift_reference() -> dict[str, Any]:
    return _load_optional(DRIFT_REFERENCE_PATH)


@lru_cache(maxsize=1)
def load_logging_power_law_reference() -> dict[str, Any]:
    return _load_optional(LOGGING_POWER_LAW_REFERENCE_PATH)


@lru_cache(maxsize=1)
def load_logging_rate_reference() -> dict[str, Any]:
    """Published logging-period results over the three measurement conditions."""

    return _load_optional(LOGGING_RATE_REFERENCE_PATH)


@lru_cache(maxsize=1)
def load_noise_lpf_reference() -> dict[str, Any]:
    """Published anti-alias-filter results: the gate, transition table, heatmap."""

    return _load_optional(NOISE_LPF_REFERENCE_PATH)


@lru_cache(maxsize=1)
def load_closed_loop_damping_reference() -> dict[str, Any]:
    """Published K_p* gain-sweep results."""

    return _load_optional(CLOSED_LOOP_DAMPING_REFERENCE_PATH)


@lru_cache(maxsize=1)
def load_retuning_reference() -> dict[str, Any]:
    """Published digital-twin retuning budget and per-method statistics."""

    return _load_optional(RETUNING_REFERENCE_PATH)


@lru_cache(maxsize=1)
def load_experiment_ledger() -> dict[str, Any]:
    return _load_optional(EXPERIMENT_LEDGER_PATH)


@lru_cache(maxsize=1)
def load_figure_index() -> dict[str, Any]:
    return _load_optional(FIGURES_PATH)
