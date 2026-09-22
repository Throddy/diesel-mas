"""Properties of the independent plant model and of the closed loop on it.

The model exists so the simulator is not the system checking itself.  These
tests pin the physics it claims, the dynamics it implements, and the fact that
it responds differently from model M.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.config import load_config, project_root
from src.sim.plant_model import PlantModel, PlantParams

CFG = load_config()
BASE = {
    "feed_sulfur_mgkg": 9343.0,
    "temperature_c": 368.15,
    "feed_tph": 214.86,
    "pressure_mpa": 3.675,
}


@pytest.fixture(scope="module")
def plant() -> PlantModel:
    return PlantModel().calibrate()


def test_calibration_reproduces_its_reference_point(plant):
    """One equation, one unknown: the reference point must come back exactly."""
    p = plant.p
    value = plant.steady_state(
        p.reference_feed_sulfur_mgkg,
        p.reference_temperature_c,
        p.reference_feed_tph,
        p.reference_pressure_mpa,
    )
    assert value == pytest.approx(p.reference_product_sulfur_mgkg, rel=1e-6)


def test_signs_follow_hydrotreating_chemistry(plant):
    """Hotter and higher pressure clean more; more feed cleans less."""
    base = plant.steady_state(**BASE)
    assert plant.steady_state(**{**BASE, "temperature_c": BASE["temperature_c"] + 5}) < base
    assert plant.steady_state(**{**BASE, "temperature_c": BASE["temperature_c"] - 5}) > base
    assert plant.steady_state(**{**BASE, "feed_tph": BASE["feed_tph"] * 1.1}) > base
    assert plant.steady_state(**{**BASE, "pressure_mpa": BASE["pressure_mpa"] * 1.05}) < base


def test_response_is_monotone_in_temperature(plant):
    values = [
        plant.steady_state(**{**BASE, "temperature_c": BASE["temperature_c"] + d})
        for d in (-6, -3, 0, 3, 6)
    ]
    assert values == sorted(values, reverse=True)


def test_plant_disagrees_with_model_m():
    """The whole point: the plant must not reproduce the optimizer's beliefs."""
    plant = PlantModel().calibrate()
    plant_sensitivity = plant.temperature_sensitivity(**BASE)
    m = json.loads(
        (project_root() / CFG.main["paths"]["models"] / "response_model.json").read_text(
            encoding="utf-8"
        )
    )["temperature_log_sensitivity_joint"]
    assert plant_sensitivity < 0 and m < 0, "both must agree on the sign"
    assert (
        abs(plant_sensitivity / m) > 1.5
    ), "plant and model M respond too similarly to test robustness"


def test_nothing_happens_before_the_transport_delay(plant):
    """A change cannot show up at the analyser sooner than the material arrives."""
    before_delay = plant.step(
        10.0, 5.0, elapsed_minutes=plant.p.delay_minutes - 10, step_minutes=10
    )
    assert before_delay == pytest.approx(10.0)


def test_after_the_delay_the_value_moves_towards_the_target(plant):
    first = plant.step(10.0, 5.0, elapsed_minutes=plant.p.delay_minutes, step_minutes=10)
    assert 5.0 < first < 10.0
    second = plant.step(first, 5.0, elapsed_minutes=plant.p.delay_minutes + 10, step_minutes=10)
    assert 5.0 < second < first


def test_relaxation_reaches_the_target_given_enough_time(plant):
    value = 10.0
    for step in range(200):
        value = plant.step(
            value, 5.0, elapsed_minutes=plant.p.delay_minutes + step * 10, step_minutes=10
        )
    assert value == pytest.approx(5.0, abs=0.05)


def test_longer_delay_postpones_the_response():
    slow = PlantModel(PlantParams(delay_minutes=720.0)).calibrate()
    assert slow.step(10.0, 5.0, elapsed_minutes=400, step_minutes=10) == pytest.approx(10.0)


def test_distorted_sensitivity_changes_the_response_not_the_level():
    """A distorted plant still starts from the same operating point."""
    from src.sim.closed_loop import SimConfig, build_default_plant

    base = build_default_plant(SimConfig())
    strong = build_default_plant(SimConfig(sensitivity_factor=1.5))
    assert strong.steady_state(**BASE) == pytest.approx(base.steady_state(**BASE), rel=1e-6)
    assert abs(strong.temperature_sensitivity(**BASE)) > abs(base.temperature_sensitivity(**BASE))


def test_delay_factor_stretches_the_delay():
    from src.sim.closed_loop import SimConfig, build_default_plant

    slow = build_default_plant(SimConfig(delay_factor=2.0))
    assert slow.p.delay_minutes == pytest.approx(SimConfig().lag_minutes * 2.0)


def test_two_identical_runs_give_identical_trajectories():
    from src.pipeline import build_system
    from src.sim.closed_loop import ClosedLoopSimulator, SimConfig

    if not (project_root() / CFG.main["paths"]["interim"] / "telemetry.parquet").exists():
        pytest.skip("canonical layer not built")
    system = build_system(CFG)
    start = pd.Timestamp(CFG.main["demo"]["quality_risk"])
    config = SimConfig(steps=3, settle_minutes=60.0)
    first = ClosedLoopSimulator(system, config).run(start, "mas")
    second = ClosedLoopSimulator(system, config).run(start, "mas")
    assert [r.sulfur_mgkg for r in first.records] == [r.sulfur_mgkg for r in second.records]


def test_advice_never_pushes_sulfur_above_its_starting_value():
    """Safety under model error: acting must not make the product worse."""
    from src.pipeline import build_system
    from src.sim.closed_loop import ClosedLoopSimulator, SimConfig

    if not (project_root() / CFG.main["paths"]["interim"] / "telemetry.parquet").exists():
        pytest.skip("canonical layer not built")
    system = build_system(CFG)
    start = pd.Timestamp(CFG.main["demo"]["limit_breach"])
    for sensitivity in (1.0, 1.5, 1 / 1.5):
        config = SimConfig(steps=6, settle_minutes=600.0, sensitivity_factor=sensitivity)
        result = ClosedLoopSimulator(system, config).run(start, "mas")
        trajectory = [r.sulfur_mgkg for r in result.records]
        assert max(trajectory) <= trajectory[0] + 1e-6, (sensitivity, max(trajectory))
