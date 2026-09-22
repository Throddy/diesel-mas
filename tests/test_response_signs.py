"""The sign constraint is the difference between configurations C and E.

The claim these tests protect: fitted the ordinary way on this closed loop, the
response model puts the wrong sign on temperature, and the shipped model M does
not.  If the data or the fit ever change that, the documents describing
configuration E stop being true, and these tests say so.
"""

import json

import numpy as np
import pytest

from src.config import load_config, project_root
from src.models.response_model import TAGS, ResponseModel

CFG = load_config()


@pytest.fixture(scope="module")
def models():
    interim = project_root() / CFG.main["paths"]["interim"]
    if not (interim / "telemetry.parquet").exists():
        pytest.skip("canonical layer not built")
    from src.data.loaders import clean_telemetry, feed_sulfur, target_series
    from src.data.store import load_df

    tel = clean_telemetry(load_df(interim / "telemetry.pkl"), CFG)
    lims = load_df(interim / "lims_long.pkl")
    target, feed = target_series(lims, CFG), feed_sulfur(lims)
    shipped = ResponseModel.fit(tel, target, feed, CFG)
    free = ResponseModel.fit(tel, target, feed, CFG, constrained=False, use_prior=False)
    return shipped, free


def test_shipped_model_keeps_every_physical_sign(models):
    shipped, _ = models
    scaled = shipped.coefficient[1:] / shipped.scale
    assert (scaled >= 0).all(), dict(zip(TAGS, scaled.tolist(), strict=False))


def test_unconstrained_fit_reverses_the_temperature_sign(models):
    """The straightforward fit disagrees with the chemistry on temperature."""
    _, free = models
    scaled = free.coefficient[1:] / free.scale
    assert scaled[0] < 0, scaled[0]
    assert free.diagnostics["temperature_log_sensitivity_joint"] > 0


def test_unconstrained_fit_keeps_the_other_three_signs(models):
    """Only temperature flips: feed rate, pressure and feed sulfur stay right."""
    _, free = models
    scaled = free.coefficient[1:] / free.scale
    assert (scaled[1:] >= 0).all(), dict(zip(TAGS[1:], scaled[1:].tolist(), strict=False))


def test_fitting_the_free_model_does_not_disturb_m(models):
    """Configuration E must not change the shipped artifact."""
    shipped, _ = models
    saved = ResponseModel.load(
        project_root() / CFG.main["paths"]["models"] / "response_model.joblib"
    )
    assert np.allclose(saved.coefficient, shipped.coefficient)


def test_direction_against_physics_is_detected():
    from scripts.arch_compare import contradicts_physics

    signs = {"ht_r201_gss_outlet_temp": ("ht:T5", -1), "ht_feed_flow_mass": ("ht:F9", 1)}
    lower_temperature = {
        "moves": {"ht_r201_gss_outlet_temp": 370.0},
        "moves_from": {"ht_r201_gss_outlet_temp": 380.0},
    }
    raise_temperature = {
        "moves": {"ht_r201_gss_outlet_temp": 382.0},
        "moves_from": {"ht_r201_gss_outlet_temp": 380.0},
    }
    raise_feed = {"moves": {"ht_feed_flow_mass": 210.0}, "moves_from": {"ht_feed_flow_mass": 200.0}}
    pair_with_one_wrong = {
        "moves": {"ht_r201_gss_outlet_temp": 382.0, "ht_feed_flow_mass": 210.0},
        "moves_from": {"ht_r201_gss_outlet_temp": 380.0, "ht_feed_flow_mass": 200.0},
    }
    assert contradicts_physics(lower_temperature, signs)
    assert not contradicts_physics(raise_temperature, signs)
    assert contradicts_physics(raise_feed, signs)
    assert contradicts_physics(pair_with_one_wrong, signs)


def test_reported_signs_match_the_saved_report():
    """The published verdict is regenerated, not typed."""
    path = project_root() / CFG.main["paths"]["reports"] / "response_unconstrained.json"
    if not path.is_file():
        pytest.skip("отчёт не построен")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["n_signs_against_physics"] == 1
    assert report["sign_verdict"]["ht:T5"] == "противоречит физике"
