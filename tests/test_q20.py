"""Guard the conclusion about ht:Q20 so a later change cannot quietly reverse it.

The tag reads at feed level but does not follow the laboratory feed sulfur, so
it is not wired in as a source.  These tests pin the evidence: if the data or
the alignment logic change, the decision gets re-examined rather than inherited.
"""

from __future__ import annotations

import json

import pytest

from src.config import load_config, project_root

CFG = load_config()
REPORT = project_root() / CFG.main["paths"]["reports"] / "q20_assessment.json"


@pytest.fixture(scope="module")
def assessment() -> dict:
    if not REPORT.exists():
        pytest.skip("run scripts/assess_q20.py first")
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_q20_reads_at_feed_level_not_product_level(assessment):
    """Three orders of magnitude separate it from the product analyser."""
    levels = assessment["levels"]
    assert levels["analyser_over_product"] > 100
    low, high = levels["analyser_p05_p95"]
    feed_low, feed_high = levels["feed_lab_p05_p95"]
    assert low < feed_high and high > feed_low, "ranges must overlap the laboratory feed"


def test_q20_does_not_track_laboratory_feed_sulfur(assessment):
    """The reason it is not used: level agrees, movement does not.

    If this ever stops failing - a corrected alignment, more analyses, a fixed
    analyser - the decision in reports/q20_assessment.json has to be revisited.
    """
    agreement = assessment["agreement_with_feed"]
    assert (
        agreement["pearson"] < 0.5
    ), "Q20 now tracks the laboratory: re-run the assessment and reconsider wiring it in"
    assert agreement["spearman"] < 0.5


def test_q20_is_not_used_as_a_feed_sulfur_source():
    """Feed sulfur is read from the laboratory series only."""
    import inspect

    from src.data import loaders

    source = inspect.getsource(loaders.feed_sulfur)
    assert "lims_series" in source, "feed sulfur must come from the LIMS series"
    assert "Q20" not in source, "the analyser is not a feed-sulfur source"
    not_controls = {entry["tag"] for entry in CFG.controls.get("not_controls", [])}
    assert "ht:Q20" in not_controls


def test_q20_health_is_recorded(assessment):
    """The sentinel share is a property of this tag and is measured, not assumed."""
    health = assessment["health"]
    assert health["sentinel_share"] > 0.10
    assert health["missing_after_masking_share"] == pytest.approx(
        health["sentinel_share"], abs=0.02
    )
