"""Tests for the expected-loss machinery and the cooldown.

The loss comparison is implemented, measured and left behind a flag: on this
unit it reproduces the threshold rule.  The cooldown is on, because it is what
actually removed the overshoot.  These tests pin both, so the flag settings
stay a decision rather than an accident.
"""

from __future__ import annotations

import json

import pytest

from src.config import load_config, project_root
from src.decision.expected_loss import (
    LossWeights,
    RecentMove,
    cooldown_block,
    evaluate,
    expected_loss,
    intervention_cost,
    off_spec_loss,
)

CFG = load_config()


class Candidate:
    """Minimal stand-in carrying the fields the pricing reads."""

    def __init__(
        self,
        action_id="A1",
        risk=0.1,
        energy=1.0,
        production=200.0,
        severity=0.3,
        magnitude=0.1,
        hold=False,
        feasible=True,
        moves=None,
    ):
        self.action_id = action_id
        self.risk_exceed_limit = risk
        self.energy_proxy = energy
        self.production_proxy = production
        self.reliability_severity = severity
        self.action_magnitude = magnitude
        self.is_do_nothing = hold
        self.feasible = feasible
        self.moves = moves or {}
        self.predicted_sulfur = 8.0
        self.predicted_sulfur_upper = 9.5
        self.notes = []
        self.rejection_reasons = []
        self.score = None


def test_holding_the_regime_costs_nothing_to_intervene():
    weights = LossWeights()
    hold = Candidate(hold=True)
    assert intervention_cost(hold, hold, weights) == 0.0


def test_only_the_excess_over_holding_is_charged():
    """Energy the regime already spends is not billed to a move that keeps it."""
    weights = LossWeights()
    hold = Candidate(hold=True, energy=1.0, severity=0.3)
    same = Candidate(energy=1.0, severity=0.3, magnitude=0.0)
    assert intervention_cost(same, hold, weights) == pytest.approx(0.0)
    costly = Candidate(energy=1.5, severity=0.4, magnitude=0.2)
    assert intervention_cost(costly, hold, weights) > 0


def test_off_spec_loss_scales_with_probability():
    weights = LossWeights(off_spec=100.0)
    assert off_spec_loss(Candidate(risk=0.1), weights) == pytest.approx(10.0)
    assert off_spec_loss(Candidate(risk=0.8), weights) == pytest.approx(80.0)


def test_probability_of_twelve_and_eighty_percent_are_priced_differently():
    """The complaint against the threshold rule, stated as a test."""
    weights = LossWeights()
    low, high = Candidate(risk=0.12), Candidate(risk=0.80)
    assert expected_loss(high, Candidate(hold=True), weights) > expected_loss(
        low, Candidate(hold=True), weights
    )


def test_missing_probability_yields_no_price():
    assert expected_loss(Candidate(risk=None), Candidate(hold=True), LossWeights()) is None


def test_a_cheap_move_against_a_large_risk_is_taken():
    weights = LossWeights(off_spec=100.0, margin=0.05)
    hold = Candidate("A00", risk=0.5, hold=True)
    move = Candidate("A01", risk=0.05, energy=1.1, magnitude=0.1)
    verdict = evaluate([hold, move], weights)
    assert verdict["acts"] is True
    assert verdict["decision"] == "A01"


def test_an_expensive_move_against_a_small_risk_is_refused():
    weights = LossWeights(off_spec=100.0, margin=0.05)
    hold = Candidate("A00", risk=0.01, hold=True)
    move = Candidate("A01", risk=0.005, energy=3.0, production=150.0, magnitude=0.8)
    verdict = evaluate([hold, move], weights)
    assert verdict["acts"] is False
    assert verdict["decision"] == "A00"


def test_margin_prevents_acting_for_a_negligible_gain():
    weights = LossWeights(off_spec=100.0, margin=0.5)
    hold = Candidate("A00", risk=0.10, hold=True)
    move = Candidate("A01", risk=0.098, energy=1.0, magnitude=0.0)
    assert evaluate([hold, move], weights)["acts"] is False


def test_a_second_move_on_the_same_control_is_held_back():
    recent = [RecentMove("ht_r201_gss_outlet_temp", minutes_ago=60, delta=1.0)]
    candidate = Candidate(moves={"ht_r201_gss_outlet_temp": 380.0})
    assert cooldown_block(candidate, recent, 180.0) is not None


def test_the_hold_expires_after_one_time_constant():
    recent = [RecentMove("ht_r201_gss_outlet_temp", minutes_ago=200, delta=1.0)]
    candidate = Candidate(moves={"ht_r201_gss_outlet_temp": 380.0})
    assert cooldown_block(candidate, recent, 180.0) is None


def test_a_different_control_is_not_held_back():
    recent = [RecentMove("ht_r201_gss_outlet_temp", minutes_ago=30, delta=1.0)]
    candidate = Candidate(moves={"ht_feed_flow_mass": 200.0})
    assert cooldown_block(candidate, recent, 180.0) is None


def test_recovery_from_off_spec_is_never_held_back():
    """Waiting is the more expensive mistake while the product is out of range."""
    recent = [RecentMove("ht_r201_gss_outlet_temp", minutes_ago=10, delta=1.0)]
    candidate = Candidate(moves={"ht_r201_gss_outlet_temp": 380.0})
    assert cooldown_block(candidate, recent, 180.0, still_off_spec=True) is None


def test_visible_effect_lifts_the_hold():
    recent = [RecentMove("ht_r201_gss_outlet_temp", minutes_ago=10, delta=1.0)]
    candidate = Candidate(moves={"ht_r201_gss_outlet_temp": 380.0})
    assert cooldown_block(candidate, recent, 180.0, effect_seen=True) is None


def test_holding_the_regime_is_never_blocked():
    recent = [RecentMove("ht_r201_gss_outlet_temp", minutes_ago=1, delta=1.0)]
    assert cooldown_block(Candidate(hold=True), recent, 180.0) is None


def test_expected_loss_rule_is_off_and_cooldown_is_on():
    """The measured outcome, kept as configuration rather than as memory."""
    section = CFG.main["expected_loss"]
    assert (
        section["enabled"] is False
    ), "the loss comparison reproduced the threshold rule; see reports/decision_rules.json"
    assert section["cooldown_minutes"] > 0, "the cooldown is what removed the overshoot"


def test_comparison_report_shows_the_two_rules_agree():
    path = project_root() / CFG.main["paths"]["reports"] / "decision_rules.json"
    if not path.exists():
        pytest.skip("run scripts/compare_decision_rules.py first")
    report = json.loads(path.read_text(encoding="utf-8"))
    threshold, expected = report["threshold_rule"], report["expected_loss_rule"]
    assert threshold["violations"] == expected["violations"] == 0
    assert threshold["action_rate"] == pytest.approx(expected["action_rate"])
