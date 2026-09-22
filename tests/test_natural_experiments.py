"""Tests for the natural-experiment search.

The search found nothing on this unit, which is a result about the data rather
than a bug.  These tests check that the search itself works: it finds a planted
step, rejects the cases it is supposed to reject, and stays deterministic.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.config import load_config, project_root
from src.eval.natural_experiments import (
    FEED,
    PRESSURE,
    TEMPERATURE,
    Episode,
    attach_sulfur,
    find_episodes,
    sensitivity,
    usable,
)

CFG = load_config()
SECTION = CFG.main["natural_experiments"]


def synthetic(
    step_c: float = 5.0, hold_hours: float = 14.0, noise: float = 0.0, feed_drift: float = 0.0
) -> tuple[pd.DataFrame, pd.Series]:
    """A clean planted step: two held levels with a ramp between them."""
    index = pd.date_range("2026-01-01", periods=int(hold_hours * 2 * 6) + 48, freq="10min")
    middle = len(index) // 2
    temperature = np.concatenate(
        [np.full(middle, 370.0), np.full(len(index) - middle, 370.0 + step_c)]
    )
    if noise:
        temperature = temperature + np.random.default_rng(0).normal(0, noise, len(index))
    frame = pd.DataFrame(
        {
            TEMPERATURE: temperature,
            FEED: np.full(len(index), 210.0) + np.linspace(0, feed_drift, len(index)),
            PRESSURE: np.full(len(index), 3.67),
        },
        index=index,
    )
    return frame, pd.Series(True, index=index)


def test_planted_step_is_found():
    frame, eligible = synthetic()
    episodes = find_episodes(frame, eligible, TEMPERATURE, SECTION)
    assert len(episodes) >= 1
    episode = episodes[0]
    assert episode.delta == pytest.approx(5.0, abs=0.2)
    assert episode.value_after > episode.value_before


def test_step_below_the_threshold_is_ignored():
    frame, eligible = synthetic(step_c=1.0)
    assert find_episodes(frame, eligible, TEMPERATURE, SECTION) == []


def test_level_that_is_not_held_is_rejected():
    """A wandering temperature is what this unit actually does, and must not count."""
    frame, eligible = synthetic(step_c=5.0, noise=2.0)
    assert find_episodes(frame, eligible, TEMPERATURE, SECTION) == []


def test_step_accompanied_by_a_feed_change_is_rejected():
    """If another control moved too, the effect cannot be attributed."""
    frame, eligible = synthetic(step_c=5.0, feed_drift=40.0)
    assert find_episodes(frame, eligible, TEMPERATURE, SECTION) == []


def test_shutdown_inside_the_window_is_rejected():
    frame, eligible = synthetic()
    middle = len(frame) // 2
    eligible.iloc[middle - 5 : middle + 5] = False
    assert find_episodes(frame, eligible, TEMPERATURE, SECTION) == []


def test_sulfur_is_read_after_the_transport_delay():
    """The comparison window opens after the lag, not at the change."""
    frame, eligible = synthetic()
    episodes = find_episodes(frame, eligible, TEMPERATURE, SECTION)
    assert episodes
    change = episodes[0].change_time
    lab = pd.Series(
        [9.0, 9.0, 6.0, 6.0],
        index=[
            change - pd.Timedelta(hours=6),
            change - pd.Timedelta(hours=2),
            change + pd.Timedelta(hours=10),
            change + pd.Timedelta(hours=14),
        ],
    )
    feed = pd.Series([9000.0], index=[change - pd.Timedelta(days=1)])
    attached = attach_sulfur(episodes, lab, None, feed, SECTION, lag_minutes=360)
    assert attached[0].sulfur_before == pytest.approx(9.0)
    assert attached[0].sulfur_after == pytest.approx(6.0)
    assert attached[0].sulfur_source == "LIMS"


def test_episode_with_a_moving_feed_sulfur_is_excluded():
    frame, eligible = synthetic()
    episodes = find_episodes(frame, eligible, TEMPERATURE, SECTION)
    change = episodes[0].change_time
    lab = pd.Series(
        [9.0, 6.0], index=[change - pd.Timedelta(hours=2), change + pd.Timedelta(hours=12)]
    )
    feed = pd.Series(
        [9000.0, 12000.0], index=[change - pd.Timedelta(days=1), change + pd.Timedelta(hours=1)]
    )
    attached = attach_sulfur(episodes, lab, None, feed, SECTION, lag_minutes=360)
    assert usable(attached) == []


def test_sensitivity_interval_is_deterministic():
    episodes = [
        Episode(
            TEMPERATURE,
            pd.Timestamp("2026-01-01"),
            pd.Timestamp("2026-01-01"),
            pd.Timestamp("2026-01-02"),
            370.0,
            375.0,
            5.0,
            sulfur_before=9.0,
            sulfur_after=7.0,
        )
        for _ in range(12)
    ]
    first, second = sensitivity(episodes), sensitivity(episodes)
    assert first == second
    assert first["median"] < 0


def test_recorded_result_is_no_episodes_on_this_unit():
    """The finding itself: this plant offers no usable natural experiment.

    If a future run finds some, the report changes and this test fails, which
    is the point: the conclusion should not be inherited silently.
    """
    path = project_root() / CFG.main["paths"]["reports"] / "natural_experiments.json"
    if not path.exists():
        pytest.skip("run scripts/natural_experiments.py first")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["conclusion"]["episodes"] == 0
    assert report["conclusion"]["sufficient"] is False
