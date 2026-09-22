"""Unit tests for the non-negotiable rules of the system."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.aliases import resolve_metric
from src.data.alignment import asof_value, assert_no_future, trailing_window
from src.data.units import canonical_unit_token, convert
from src.data.validation import causal_run_length, find_flatlines
from src.models.uncertainty import ConformalCalibration
from src.optimization.constraints import all_pass, check_candidate
from src.schemas import BlendConstraint, CandidateAction

CFG = load_config()


def test_asof_is_backward_only():
    s = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.to_datetime(["2024-01-01 00:00", "2024-01-01 06:00", "2024-01-01 12:00"]),
    )
    v, ts, age = asof_value(s, pd.Timestamp("2024-01-01 07:00"))
    assert v == 2.0 and ts == pd.Timestamp("2024-01-01 06:00") and age == 60.0
    assert asof_value(s, pd.Timestamp("2023-12-31"))[0] is None


def test_trailing_window_excludes_future():
    idx = pd.date_range("2024-01-01", periods=10, freq="10min")
    df = pd.DataFrame({"x": range(10)}, index=idx)
    w = trailing_window(df, idx[5], 30)
    assert w.index.max() == idx[5] and len(w) == 3


def test_assert_no_future_raises():
    t = pd.Series(pd.to_datetime(["2024-01-02"]))
    cutoff = pd.Series(pd.to_datetime(["2024-01-01"]))
    with pytest.raises(AssertionError):
        assert_no_future(t, cutoff)
    assert_no_future(cutoff, t)


def test_causal_run_length_is_causal():
    s = pd.Series([1, 1, 1, 2], index=pd.date_range("2024-01-01", periods=4, freq="10min"))
    r = causal_run_length(s)
    assert list(r) == [1, 2, 3, 1]


def test_source_priority_lims_over_pak():
    src = {"lims": {"value": 22.0, "health": "OK"}, "pak": {"value": 7.7, "health": "OK"}}
    choice = "LIMS" if src["lims"]["value"] is not None and src["lims"]["health"] == "OK" else "PAK"
    assert choice == "LIMS"


def test_staleness_ratio_marks_old_lims():
    from src.data.validation import inter_arrival_stats, staleness_ratio

    idx = pd.date_range("2024-01-01", periods=50, freq="1D")
    stats = inter_arrival_stats(idx)
    assert staleness_ratio(1440, stats) == pytest.approx(1.0)
    assert staleness_ratio(5760, stats) > 1.0


def test_flatline_detection_finds_stuck_analyzer():
    idx = pd.date_range("2024-03-16 18:20", periods=200, freq="10min")
    s = pd.Series([7.70516] * 150 + list(np.linspace(7.7, 9.0, 50)), index=idx)
    segs = find_flatlines(s, min_points=36)
    assert segs and segs[0].n_points >= 150
    assert segs[0].value == pytest.approx(7.70516)


def test_mg_sulfur_alias_resolution():
    assert resolve_metric("24-2000:Mg.Sulfur") == "Mg.Sulfur"
    assert resolve_metric("24-2000:Mg.Sulfur.Q") == "Mg.Sulfur"
    assert resolve_metric("Mg.Sulfur.Q") == "Mg.Sulfur"
    assert resolve_metric("totally unknown tag") is None


def test_nonnumeric_lims_values_are_flagged_not_silently_dropped():
    from src.config import project_root
    from src.data.store import load_df

    path = project_root() / CFG.main["paths"]["interim"] / "lims_long.parquet"
    if not path.exists():
        pytest.skip("canonical layer not built")
    lims = load_df(path)
    flagged = lims[lims["data_quality_flags"].str.contains("NONNUMERIC", na=False)]
    assert len(flagged) > 0
    assert flagged["value"].isna().all()


def test_unit_conversion_registry():
    assert canonical_unit_token("ppm") == "ppm_mass"
    v, assumption, ok = convert(7.7, "ppm_mass", "mg_per_kg")
    assert ok and v == 7.7 and assumption == "A-03"
    v, a, ok = convert(0.1, "mass_pct", "mg_per_kg")
    assert ok and v == pytest.approx(1000.0)
    assert convert(1.0, "deg_c", "mg_per_kg")[2] is False


def test_hard_sulfur_constraint_uses_upper_bound():
    cand = CandidateAction(
        "t",
        None,
        [],
        None,
        None,
        0.0,
        "hold",
        is_do_nothing=True,
        predicted_sulfur=9.5,
        predicted_sulfur_upper=12.0,
    )
    checks = check_candidate(cand, cfg=CFG)
    hc01 = [c for c in checks if c.constraint_id == "HC-01"][0]
    assert hc01.status == "FAIL"
    cand.predicted_sulfur_upper = 8.0
    assert [c for c in check_candidate(cand, cfg=CFG) if c.constraint_id == "HC-01"][
        0
    ].status == "PASS"


def test_conformal_upper_bound_is_one_sided_and_monotone():
    cal = ConformalCalibration(residuals=np.random.default_rng(0).normal(0, 0.2, 500), alpha=0.1)
    assert cal.upper_quantile() < cal.quantile()
    risk_low = cal.exceedance_risk(np.log(np.array([5.0])), 10.0)[0]
    risk_high = cal.exceedance_risk(np.log(np.array([9.9])), 10.0)[0]
    assert risk_high > risk_low


def test_control_support_bounds_reject_out_of_range_moves():
    cand = CandidateAction(
        "t",
        "ht_r201_gss_outlet_temp",
        ["ht:T5"],
        370.0,
        500.0,
        130.0,
        "increase",
        predicted_sulfur=5.0,
        predicted_sulfur_upper=6.0,
    )
    checks = check_candidate(cand, cfg=CFG, bounds={"ht_r201_gss_outlet_temp": (350.0, 390.0)})
    assert [c for c in checks if c.constraint_id == "HC-03"][0].status == "FAIL"
    assert not all_pass(checks)


def test_safety_veto_cannot_be_overridden():
    from src.agents.safety_agent import SafetyAgent
    from src.schemas import (
        DataQualityAssessment,
        ProcessState,
        QualityAssessment,
        ReliabilityAssessment,
        SourceStatus,
    )

    t = pd.Timestamp("2024-04-23 06:30")
    dq = DataQualityAssessment(
        decision_timestamp=t,
        sources=[
            SourceStatus("LIMS", "Mg.Sulfur", t, 2120.0, "мг/кг", 4000.0, 2.8, "SUSPECT", []),
            SourceStatus("PAK", "Mg.Sulfur", t, 7.705, "мг/кг", 0.0, None, "FAILED", []),
        ],
        flags=["PAK_FLATLINE", "SOURCE_CONFLICT"],
        telemetry_missing_share=0.0,
        telemetry_stale_tags=[],
        duplicate_timestamps=0,
        nonnumeric_values=0,
        source_disagreement={
            "lims": 2120.0,
            "pak": 7.705,
            "abs": 2112.3,
            "rel": 0.99,
            "severe": True,
            "resolution": "LIMS wins",
        },
        critical_data_ok=False,
    )
    state = ProcessState(
        timestamp=t, telemetry={}, controls={}, quality_sources={}, data_quality=dq
    )
    quality = QualityAssessment(
        t, "Mg.Sulfur", "мг/кг", 5.0, 4.0, 6.0, 0.0, 10.0, "v", confidence=0.9
    )
    rel = ReliabilityAssessment(t, 0.2, "NORMAL", 0.1, False, [], True, [], "ok")
    perfect = CandidateAction(
        "A00_do_nothing",
        None,
        [],
        None,
        None,
        0.0,
        "hold",
        is_do_nothing=True,
        predicted_sulfur=5.0,
        predicted_sulfur_upper=6.0,
    )
    out = SafetyAgent(cfg=CFG).run([perfect], state, quality, rel)
    assert out["veto"], "a failed analyser + severe source conflict must produce a veto"
    assert perfect.feasible is True
    assert out["admissible_ids"] == []
    verdict = out["verdicts"][0]
    assert verdict.admissible is False
    assert verdict.dimension == "safety"
    from src.mas.policies import resolve_conflicts

    resolved = resolve_conflicts([perfect], out["verdicts"])[0]
    assert resolved["admissible"] is False
    assert resolved["blocked_by"] == "SafetyAgent"


def test_blend_fractions_must_sum_to_one():
    bc = BlendConstraint(component_names=["A", "B"])
    assert bc.check({"A": 0.6, "B": 0.4}).status == "PASS"
    assert bc.check({"A": 0.6, "B": 0.3}).status == "FAIL"
    assert bc.check({"A": 1.2, "B": -0.2}).status == "FAIL"
    assert bc.check({"A": 1.0}).status == "FAIL"


def test_no_feasible_candidate_leads_to_abstention():
    cand = CandidateAction(
        "A00_do_nothing",
        None,
        [],
        None,
        None,
        0.0,
        "hold",
        is_do_nothing=True,
        predicted_sulfur=14.0,
        predicted_sulfur_upper=18.0,
    )
    checks = check_candidate(cand, cfg=CFG)
    cand.feasible = all_pass(checks)
    from src.agents.optimization_agent import rank_feasible

    assert cand.feasible is False
    assert rank_feasible([cand], CFG.sulfur_limit) == []


def test_do_nothing_always_present_and_first():
    from src.optimization.candidates import generate_candidates

    tel = {
        "ht:T5": 370.0,
        "ht:F26": 256.0,
        "ht:F15": 3398.0,
        "ht:F9": 218.0,
        "ht:P3": 3.65,
        "avt:T55": 381.0,
        "avt:P23": 1.11,
        "avt:F30": 127.0,
        "avt:F29": 8.5,
        "avt:F7": 265.0,
        "avt:F8": 262.0,
        "avt:F9": 209.0,
    }
    a = generate_candidates(tel, CFG)
    b = generate_candidates(tel, CFG)
    assert a[0].is_do_nothing
    assert [c.action_id for c in a] == [c.action_id for c in b]
    assert [c.proposed_value for c in a] == [c.proposed_value for c in b]
