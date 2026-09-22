"""Metamorphic and closed-network tests.

A metamorphic test does not need a ground-truth label: it states how the
output MUST change when the input changes in a known direction.  These encode
the physics the terms of reference care about, so a model that fits the data
but breaks the chemistry fails here.
"""

from __future__ import annotations

import socket

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import load_config, project_root
from src.models.response_model import ResponseModel

CFG = load_config()
MODEL_PATH = project_root() / CFG.main["paths"]["models"] / "response_model.joblib"
PHYSICS = yaml.safe_load((project_root() / "config" / "physics.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def response() -> ResponseModel:
    if not MODEL_PATH.exists():
        pytest.skip("response model not trained")
    return ResponseModel.load(MODEL_PATH)


def _mid_point(model: ResponseModel) -> dict:
    """A state in the middle of the model support, so nudges stay inside it."""
    return {tag: float(np.mean(bounds)) for tag, bounds in model.bounds.items()}


def _nudge(model: ResponseModel, tag: str, share_of_support: float) -> tuple[float, float]:
    """Move one variable by a share of its support width, staying inside it."""
    before = _mid_point(model)
    low, high = model.bounds[tag]
    after = dict(before)
    after[tag] = before[tag] + share_of_support * (high - low)
    assert low <= after[tag] <= high
    return model.effect(before, after)


def test_higher_reactor_temperature_never_raises_predicted_sulfur(response):
    """R-MEET-11 / physics.yaml: hotter reactor means deeper desulfurisation."""
    before = _mid_point(response)
    after = dict(before)
    after["ht:T5"] = before["ht:T5"] + 5.0
    central, pessimistic = response.effect(before, after)
    assert central < 0, f"d ln S for +5 °C must be negative, got {central}"
    assert pessimistic <= 0, f"even the pessimistic bound must not raise sulfur: {pessimistic}"


def test_temperature_effect_is_monotone_in_step_size(response):
    """A bigger temperature step must not give a smaller sulfur reduction."""
    before = _mid_point(response)
    effects = []
    for step in (1.0, 3.0, 6.0):
        after = dict(before)
        after["ht:T5"] = before["ht:T5"] + step
        effects.append(response.effect(before, after)[0])
    assert effects[0] >= effects[1] >= effects[2], effects


def test_higher_feed_rate_never_lowers_predicted_sulfur(response):
    """Less residence time per unit of feed leaves more sulfur (physics.yaml)."""
    central, _ = _nudge(response, "ht:F9", 0.05)
    assert central > 0, central


def test_higher_feed_sulfur_never_lowers_predicted_sulfur(response):
    """Feed sulfur is a disturbance with a positive sign (R-MEET-07, K-50)."""
    central, _ = _nudge(response, "feed_sulfur_mgkg", 0.10)
    assert central > 0, central


def test_higher_pressure_never_raises_predicted_sulfur(response):
    central, _ = _nudge(response, "ht:P3", 0.05)
    assert central <= 0, central


def test_fitted_signs_agree_with_declared_physics(response):
    """Every coefficient sign in the fitted model matches config/physics.yaml."""
    for tag, coefficient in response.diagnostics["coefficients"].items():
        declared = PHYSICS["variables"][tag]["sulfur_sign"]
        if abs(coefficient) < 1e-12:
            continue
        transformed_sign = 1 if coefficient > 0 else -1
        assert transformed_sign == 1, (
            f"{tag}: the design matrix is built so every coefficient is >= 0; " f"got {coefficient}"
        )
        assert declared in (-1, 1), tag


def test_declared_temperature_sensitivity_is_negative(response):
    """The published d ln S / dT must be negative whichever estimate is used."""
    diagnostics = response.diagnostics
    assert diagnostics["temperature_log_sensitivity_joint"] < 0
    assert diagnostics["temperature_log_sensitivity_prior"] < 0
    low, high = diagnostics["temperature_sensitivity_bootstrap"]
    assert high < 0, (low, high)


def test_move_outside_model_support_is_refused_not_extrapolated(response):
    """Extrapolation is forbidden: the model raises instead of guessing."""
    before = _mid_point(response)
    after = dict(before)
    after["ht:T5"] = response.bounds["ht:T5"][1] + 50.0
    with pytest.raises(ValueError):
        response.effect(before, after)


def test_system_runs_without_any_network_access(monkeypatch):
    """The deployment target has no internet, so nothing may open a connection.

    Imports are resolved before the sockets are taken away.  Replacing
    ``socket.socket`` with a function breaks any later lazy import of ``ssl``,
    which subclasses it, and that failure says nothing about network use.
    """
    import ssl  # noqa: F401  resolve the lazy import while sockets still work

    from src.pipeline import build_system

    interim = project_root() / CFG.main["paths"]["interim"]
    if not (interim / "telemetry.parquet").exists():
        pytest.skip("canonical layer not built")

    def refuse(*args, **kwargs):
        raise OSError("network access is not allowed in a closed-loop deployment")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    system = build_system(CFG)
    recommendation = system.decide(pd.Timestamp(CFG.main["demo"]["stable"]), log=False)
    assert recommendation.headline
    assert recommendation.decision_timestamp is not None


TABS = (
    "Обзор",
    "Качество",
    "Горизонты",
    "Качество данных",
    "Надёжность",
    "Рекомендация",
    "Альтернативы и Парето",
    "Трассировка агентов",
    "Обмен сообщениями",
)

ABSTENTION = "Надёжной рекомендации нет"


def _dashboard(scenario=None):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(project_root() / "app.py"), default_timeout=900)
    app.run()
    if scenario is not None:
        box = next(b for b in app.selectbox if "stable" in (b.options or []))
        box.select(scenario).run()
    return app


def _assert_healthy(app, scenario):
    assert not app.exception, (scenario, [e.value for e in app.exception])
    unexpected = [e.value for e in app.error if ABSTENTION not in str(e.value)]
    assert not unexpected, (scenario, unexpected)


def test_dashboard_renders_every_tab_without_exceptions():
    """The operator UI must actually run: docs once claimed it was never started."""
    pytest.importorskip("streamlit")
    interim = project_root() / CFG.main["paths"]["interim"]
    if not (interim / "telemetry.parquet").exists():
        pytest.skip("canonical layer not built")
    app = _dashboard()
    _assert_healthy(app, "по умолчанию")
    assert [t.label for t in app.tabs] == list(TABS)
    empty = [t.label for t in app.tabs if not list(t.children)]
    assert not empty, f"вкладки без содержимого: {empty}"


@pytest.mark.parametrize("scenario", ["stable", "quality_risk", "bad_data"])
def test_dashboard_runs_every_scenario(scenario):
    """Switching the scenario must not break any tab, abstention included."""
    pytest.importorskip("streamlit")
    interim = project_root() / CFG.main["paths"]["interim"]
    if not (interim / "telemetry.parquet").exists():
        pytest.skip("canonical layer not built")
    app = _dashboard(scenario)
    _assert_healthy(app, scenario)
    assert len(app.tabs) == len(TABS)


def test_evaluation_report_holds_no_wall_clock_values():
    """Two runs must produce the same bytes, so timings live in their own file.

    Wall-clock numbers measure the machine, not the system; keeping them in the
    report made every pair of runs differ and hid real non-determinism.
    """
    import json

    report = project_root() / CFG.main["paths"]["reports"] / "evaluation_report.json"
    if not report.is_file():
        pytest.skip("отчёт не построен")
    text = report.read_text(encoding="utf-8")
    for field in (
        "inference_seconds_mean",
        "inference_seconds_max",
        "seconds_per_cycle_mean",
        "train_seconds",
    ):
        assert field not in text, field
    timing = report.with_name("evaluation_timing.json")
    assert timing.is_file(), "замеры времени должны писаться в evaluation_timing.json"
    assert "inference_seconds_mean" in json.loads(timing.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def system():
    interim = project_root() / CFG.main["paths"]["interim"]
    if not (interim / "telemetry.parquet").exists():
        pytest.skip("canonical layer not built")
    from src.pipeline import build_system

    return build_system(CFG)


def test_no_recommended_action_exceeds_the_sulfur_limit(system):
    """The hard constraint holds on every demo moment: upper bound <= 10 mg/kg."""
    for scenario in ("stable", "quality_risk", "bad_data"):
        rec = system.decide(pd.Timestamp(CFG.main["demo"][scenario]), log=False)
        action = rec.selected_action
        if not action or action.get("is_do_nothing"):
            continue
        upper = action["predicted_sulfur_upper"]
        assert upper is not None and upper <= CFG.sulfur_limit, (scenario, upper)


def test_failed_analyser_with_stale_lab_result_forces_abstention(system):
    """Bad-data moment: no healthy sulfur source means no recommendation."""
    rec = system.decide(pd.Timestamp(CFG.main["demo"]["bad_data"]), log=False)
    assert rec.abstained
    assert rec.reasons, "an abstention must name its reason"


def test_no_action_outside_running_mode(system):
    """Shutdown and transient regimes never produce a control move."""
    from src.data.operating_mode import operating_modes

    modes = operating_modes(system.telemetry, CFG)
    stopped = modes.index[modes["mode"] == "SHUTDOWN"]
    if not len(stopped):
        pytest.skip("no shutdown period in the package")
    rec = system.decide(stopped[len(stopped) // 2], log=False)
    action = rec.selected_action
    assert rec.abstained or not action or action.get("is_do_nothing"), rec.headline


def test_closed_loop_simulation_is_reproducible_on_one_seed(system):
    """Same seed, same start, same trajectory - the run is a deterministic experiment."""
    from src.sim.closed_loop import ClosedLoopSimulator, SimConfig

    start = pd.Timestamp(CFG.main["demo"]["quality_risk"])
    config = SimConfig(steps=3, settle_minutes=60.0, seed=42)
    first = ClosedLoopSimulator(system, config).run(start, "mas")
    second = ClosedLoopSimulator(system, config).run(start, "mas")
    assert [r.sulfur_mgkg for r in first.records] == [r.sulfur_mgkg for r in second.records]
    assert first.summary(CFG.sulfur_limit) == second.summary(CFG.sulfur_limit)


def test_acting_policy_ends_below_holding_policy(system):
    """Advice that raises temperature must not end with more sulfur than holding."""
    from src.sim.closed_loop import ClosedLoopSimulator, SimConfig

    start = pd.Timestamp(CFG.main["demo"]["quality_risk"])
    simulator = ClosedLoopSimulator(system, SimConfig(steps=6, settle_minutes=600.0))
    holding = simulator.run(start, "do_nothing").records[-1].sulfur_mgkg
    advised = simulator.run(start, "mas").records[-1].sulfur_mgkg
    assert advised <= holding + 1e-9, (advised, holding)
