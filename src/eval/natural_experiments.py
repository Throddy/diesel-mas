"""Find episodes where operators changed the regime, and measure what followed.

Model M carries a physical prior because a closed control loop hides the effect
of the levers in ordinary data.  A prior is an assumption, though, and the only
way to check it against this unit is to find the moments when someone did move
a control and nothing else moved with it.

An episode is a step in one control, held long enough for the effect to reach
the product, with the other controls and the feed sulfur quiet.  The selection
thresholds live in config and were fixed before any effect was computed (A-26):
choosing them afterwards would turn this into a search for a desired answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

TEMPERATURE = "ht:T5"
FEED = "ht:F9"
PRESSURE = "ht:P3"


@dataclass
class Episode:
    """One operator-made step and the sulfur measured around it."""

    control: str
    start: pd.Timestamp
    change_time: pd.Timestamp
    end: pd.Timestamp
    value_before: float
    value_after: float
    delta: float
    sulfur_before: Optional[float] = None
    sulfur_after: Optional[float] = None
    sulfur_source: str = "none"
    n_before: int = 0
    n_after: int = 0
    feed_sulfur_before: Optional[float] = None
    feed_sulfur_after: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    @property
    def observed_log_change(self) -> Optional[float]:
        if not self.sulfur_before or not self.sulfur_after:
            return None
        if self.sulfur_before <= 0 or self.sulfur_after <= 0:
            return None
        return float(np.log(self.sulfur_after) - np.log(self.sulfur_before))

    def to_dict(self) -> dict:
        data = {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in self.__dict__.items()}
        data["observed_log_change"] = self.observed_log_change
        return data


def _is_stable(window: pd.Series, tolerance: float) -> bool:
    """A control counts as quiet when its range stays inside the tolerance."""
    values = window.dropna()
    if values.empty:
        return False
    return bool(float(values.max() - values.min()) <= tolerance)


def find_episodes(
    telemetry: pd.DataFrame, eligible: pd.Series, control: str, cfg_section: dict
) -> list[Episode]:
    """Scan for held steps in one control with everything else quiet.

    The scan walks candidate change points on a coarse grid, so two overlapping
    windows cannot both claim the same step.
    """
    hold = pd.Timedelta(hours=float(cfg_section["hold_hours"]))
    transition = pd.Timedelta(hours=float(cfg_section.get("transition_hours", 2.0)))
    if control == TEMPERATURE:
        min_change = float(cfg_section["temperature_change_min_c"])
        hold_tolerance = float(cfg_section["hold_tolerance_temperature_c"])
        others = {
            FEED: float(cfg_section["stable_other_feed_tph"]),
            PRESSURE: float(cfg_section["stable_other_pressure_mpa"]),
        }
    else:
        min_change = float(cfg_section["feed_change_min_tph"])
        hold_tolerance = float(cfg_section["hold_tolerance_feed_tph"])
        others = {
            TEMPERATURE: float(cfg_section["stable_other_temperature_c"]),
            PRESSURE: float(cfg_section["stable_other_pressure_mpa"]),
        }

    series = telemetry[control]
    episodes: list[Episode] = []
    cursor = telemetry.index.min() + hold
    last_end = telemetry.index.min()
    step = pd.Timedelta(hours=1)

    while cursor + hold <= telemetry.index.max():
        if cursor < last_end:
            cursor += step
            continue
        before_window = series.loc[cursor - transition - hold : cursor - transition]
        after_window = series.loc[cursor + transition : cursor + transition + hold]
        if before_window.empty or after_window.empty:
            cursor += step
            continue

        value_before = float(before_window.median())
        value_after = float(after_window.median())
        delta = value_after - value_before
        if abs(delta) < min_change:
            cursor += step
            continue
        if not (
            _is_stable(before_window, hold_tolerance) and _is_stable(after_window, hold_tolerance)
        ):
            cursor += step
            continue
        span = slice(cursor - transition - hold, cursor + transition + hold)
        if not bool(eligible.loc[span].all()):
            cursor += step
            continue
        quiet = all(
            _is_stable(telemetry[tag].loc[span], tolerance) for tag, tolerance in others.items()
        )
        if not quiet:
            cursor += step
            continue

        episodes.append(
            Episode(
                control=control,
                start=cursor - transition - hold,
                change_time=cursor,
                end=cursor + transition + hold,
                value_before=value_before,
                value_after=value_after,
                delta=float(delta),
            )
        )
        last_end = cursor + transition + hold
        cursor += hold
    return episodes


def attach_sulfur(
    episodes: list[Episode],
    laboratory: pd.Series,
    analyser: Optional[pd.Series],
    feed_sulfur: pd.Series,
    cfg_section: dict,
    lag_minutes: float,
) -> list[Episode]:
    """Measure product sulfur before and after each step, honouring the lag.

    The laboratory is preferred; where it has nothing to say in a window the
    analyser is used instead and the episode records which source it used.  The
    comparison window opens ``settle_hours`` after the change, because material
    already in the reactor still carries the old regime.
    """
    settle = pd.Timedelta(hours=float(cfg_section["settle_hours"]))
    width = pd.Timedelta(hours=float(cfg_section["compare_window_hours"]))
    lag = pd.Timedelta(minutes=float(lag_minutes))
    max_feed_change = float(cfg_section["feed_sulfur_max_relative_change"])

    for episode in episodes:
        before_from, before_to = episode.change_time - width, episode.change_time
        after_from = episode.change_time + lag + settle
        after_to = after_from + width

        lab_before = laboratory.loc[before_from:before_to].dropna()
        lab_after = laboratory.loc[after_from:after_to].dropna()
        if len(lab_before) and len(lab_after):
            episode.sulfur_before = float(lab_before.median())
            episode.sulfur_after = float(lab_after.median())
            episode.sulfur_source = "LIMS"
            episode.n_before, episode.n_after = len(lab_before), len(lab_after)
        elif analyser is not None:
            pak_before = analyser.loc[before_from:before_to].dropna()
            pak_after = analyser.loc[after_from:after_to].dropna()
            if len(pak_before) and len(pak_after):
                episode.sulfur_before = float(pak_before.median())
                episode.sulfur_after = float(pak_after.median())
                episode.sulfur_source = "PAK"
                episode.n_before, episode.n_after = len(pak_before), len(pak_after)
                episode.notes.append("лаборатория молчит в окне: использован поточный анализатор")

        feed_before = feed_sulfur.loc[: episode.change_time].tail(1)
        feed_after = feed_sulfur.loc[: episode.end].tail(1)
        if len(feed_before):
            episode.feed_sulfur_before = float(feed_before.iloc[0])
        if len(feed_after):
            episode.feed_sulfur_after = float(feed_after.iloc[0])
        if episode.feed_sulfur_before and episode.feed_sulfur_after:
            relative = abs(episode.feed_sulfur_after / episode.feed_sulfur_before - 1.0)
            if relative > max_feed_change:
                episode.notes.append(
                    f"сера сырья изменилась на {relative:.0%}: эпизод исключён из оценки"
                )
    return episodes


def usable(episodes: list[Episode]) -> list[Episode]:
    """Episodes with a measured effect and a quiet feed."""
    return [
        e
        for e in episodes
        if e.observed_log_change is not None and not any("исключён" in note for note in e.notes)
    ]


def sensitivity(episodes: list[Episode]) -> dict:
    """Effect per unit of control change, with a bootstrap interval."""
    ratios = np.array(
        [
            e.observed_log_change / e.delta
            for e in episodes
            if e.delta != 0 and e.observed_log_change is not None
        ]
    )
    if len(ratios) == 0:
        return {"n": 0}
    rng = np.random.default_rng(42)
    draws = np.array(
        [np.median(rng.choice(ratios, len(ratios), replace=True)) for _ in range(2000)]
    )
    return {
        "n": int(len(ratios)),
        "median": float(np.median(ratios)),
        "mean": float(np.mean(ratios)),
        "ci90": [float(np.quantile(draws, 0.05)), float(np.quantile(draws, 0.95))],
        "share_negative": float(np.mean(ratios < 0)),
    }


def criterion_map(
    telemetry: pd.DataFrame,
    eligible: pd.Series,
    control: str,
    cfg_section: dict,
    holds=(3, 6, 12),
    tolerances=(1.5, 3.0, 5.0),
) -> list[dict]:
    """Count episodes under other hold lengths and tolerances.

    A zero result invites the question "did the thresholds make it zero?".  The
    map answers it without moving them: it shows how the count depends on the
    criterion, so the reader can see that the shortage is in the data rather
    than in the choice.  Holds shorter than the transport delay are listed but
    marked, because the effect cannot have arrived within them.
    """
    lag_hours = float(cfg_section.get("effect_lag_hours", 6.0))
    rows = []
    for hold_hours in holds:
        for tolerance in tolerances:
            section = dict(cfg_section)
            section["hold_hours"] = hold_hours
            if control == TEMPERATURE:
                section["hold_tolerance_temperature_c"] = tolerance
            else:
                section["hold_tolerance_feed_tph"] = tolerance
            found = find_episodes(telemetry, eligible, control, section)
            rows.append(
                {
                    "hold_hours": hold_hours,
                    "tolerance": tolerance,
                    "episodes": len(found),
                    "shorter_than_transport_delay": bool(hold_hours < lag_hours),
                }
            )
    return rows
