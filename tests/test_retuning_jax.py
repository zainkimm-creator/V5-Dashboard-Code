"""The JAX port must agree with the NumPy reference it replaces.

A faster simulator that quietly disagrees is worse than no simulator, so these
tests pin the two implementations together on every campaign plant. They skip
cleanly when jax is not installed, because the CPU path is the reference and
must stay runnable without it.
"""

from __future__ import annotations

import numpy as np
import pytest

import backend.validation.retuning as R
from backend.validation.plants import parameters_for_plant

jax = pytest.importorskip("jax", reason="jax not installed; NumPy path is the reference")

from backend.validation.retuning_jax import JaxEvaluator  # noqa: E402

PLANTS = ["P01", "P02", "P03", "P06", "P09", "P10"]
# Double precision is required for agreement; jax defaults to float32, which
# would show up here as a ~1e-7 mismatch rather than a silent wrong answer.
pytestmark = pytest.mark.skipif(
    not jax.config.jax_enable_x64,
    reason="needs JAX_ENABLE_X64=1; float32 cannot match the NumPy reference",
)


def _evaluator(params, line_speed: float) -> JaxEvaluator:
    # Built from the campaign's own schedule, so the kernel scores exactly the
    # record `evaluate_gains` runs.
    return JaxEvaluator.for_campaign(params, line_speed)


@pytest.mark.parametrize("dashboard_id", PLANTS)
@pytest.mark.parametrize("kp,ti_scale", [(10.0, 1.0), (50.0, 0.2), (3.0, 5.0)])
def test_matches_numpy_cost(dashboard_id: str, kp: float, ti_scale: float) -> None:
    params, _ = parameters_for_plant(dashboard_id)
    line_speed = float(params.feeder_velocity_m_s)
    ti = ti_scale * R.plant_auto_ti_s(params, line_speed)

    reference = R.evaluate_gains(params, kp, ti, line_speed_m_s=line_speed)
    got = _evaluator(params, line_speed).evaluate(np.array([kp]), np.array([ti]))[0]

    assert got[0] == pytest.approx(reference.S, rel=2e-4)
    assert got[1] == pytest.approx(reference.rmse_y_N, rel=2e-4)
    assert got[2] == pytest.approx(reference.overshoot_percent, rel=2e-4, abs=1e-6)
    assert got[3] == pytest.approx(reference.settling_time_s, abs=2e-3)


@pytest.mark.parametrize("dashboard_id", ["P01", "P10"])
def test_batch_matches_one_at_a_time(dashboard_id: str) -> None:
    """Batching must not change any individual result."""

    params, _ = parameters_for_plant(dashboard_id)
    line_speed = float(params.feeder_velocity_m_s)
    auto_ti = R.plant_auto_ti_s(params, line_speed)
    rng = np.random.default_rng(0)
    kp = rng.uniform(1.0, 200.0, 32)
    ti = rng.uniform(0.1, 20.0, 32) * auto_ti

    evaluator = _evaluator(params, line_speed)
    batched = evaluator.cost_only(kp, ti)
    singly = np.array([evaluator.cost_only(kp[i:i + 1], ti[i:i + 1])[0]
                       for i in range(len(kp))])
    np.testing.assert_allclose(batched, singly, rtol=1e-12)


def test_batch_matches_numpy_across_a_grid() -> None:
    """The case that matters: a grid-search-sized batch, against the reference."""

    params, _ = parameters_for_plant("P03")
    line_speed = float(params.feeder_velocity_m_s)
    auto_ti = R.plant_auto_ti_s(params, line_speed)
    kp = np.array([2.0, 7.0, 20.0, 60.0, 150.0])
    ti = np.array([0.05, 0.3, 1.0, 4.0, 30.0]) * auto_ti

    got = _evaluator(params, line_speed).cost_only(kp, ti)
    want = np.array([R.evaluate_gains(params, float(k), float(t),
                                      line_speed_m_s=line_speed).S
                     for k, t in zip(kp, ti)])
    np.testing.assert_allclose(got, want, rtol=2e-4)


def test_gpu_env_helper_puts_wheel_libs_first() -> None:
    """The CUDA 13 system tree shadows the wheels' CUDA 12 libs if it wins."""

    from backend.validation.retuning_jax import configure_gpu_env, nvidia_lib_dirs

    dirs = nvidia_lib_dirs()
    if not dirs:
        pytest.skip("no pip-installed nvidia libraries in this interpreter")
    path = configure_gpu_env().split(":")
    assert path[0] == dirs[0]
    assert "/usr/lib64" in path
    assert not any("/usr/local/cuda" in p for p in path)
