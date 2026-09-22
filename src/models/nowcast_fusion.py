"""Kalman fusion of laboratory and analyser sulfur into one nowcast.

The state is log sulfur following a random walk.  Two sensors observe it:

* the online analyser (ПАК) sees the state plus a slowly wandering bias, so
  its own drift is part of the state vector and is estimated, not assumed
  away;
* the laboratory (ЛИМС) sees the state with small noise and no bias, which is
  what "the laboratory is the control fact" means in the terms of reference.

Source priority therefore stops being a rule in code.  It follows from the
measurement variances: a healthy analyser contributes, a failed one is
admitted with infinite variance and changes nothing, and the uncertainty of
the estimate grows on its own while no trustworthy measurement arrives.

A laboratory result refers to the moment the sample was taken, not to the
moment it was reported.  ``NowcastFusion`` therefore replays from the sample
time forward when a delayed analysis arrives, so a late result lands where it
belongs on the timeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.config import load_config

LOG_FLOOR = 1e-6


@dataclass
class FusionParams:
    """Noise parameters; every one of them is recorded as an assumption."""

    q_state: float = 1.9e-4
    q_drift: float = 1.9e-5
    r_lims: float = 0.0115
    r_pak: float = 0.115
    p0_state: float = 0.25
    p0_drift: float = 0.05
    step_minutes: float = 10.0

    @classmethod
    def from_config(cls, cfg=None) -> "FusionParams":
        cfg = cfg or load_config()
        section = dict(cfg.main.get("fusion", {}))
        known = {f: section[f] for f in cls.__dataclass_fields__ if f in section}
        return cls(**known)


@dataclass
class FusionState:
    """Filter state: log sulfur and the analyser's bias, with covariance."""

    x: np.ndarray = field(default_factory=lambda: np.zeros(2))
    P: np.ndarray = field(default_factory=lambda: np.eye(2))
    t: Optional[pd.Timestamp] = None

    def copy(self) -> "FusionState":
        return FusionState(self.x.copy(), self.P.copy(), self.t)


class NowcastFusion:
    """Two-state Kalman filter over the log sulfur of the hydrotreated product."""

    STATE, DRIFT = 0, 1

    def __init__(self, params: Optional[FusionParams] = None):
        self.p = params or FusionParams()

    def initial(self, value: float, t: pd.Timestamp) -> FusionState:
        """Start from one measurement, with no knowledge of the drift."""
        x = np.array([np.log(max(float(value), LOG_FLOOR)), 0.0])
        P = np.diag([self.p.p0_state, self.p.p0_drift])
        return FusionState(x, P, pd.Timestamp(t))

    def predict(self, state: FusionState, until: pd.Timestamp) -> FusionState:
        """Propagate to ``until``; uncertainty grows with elapsed time."""
        until = pd.Timestamp(until)
        if state.t is None:
            return FusionState(state.x.copy(), state.P.copy(), until)
        steps = max((until - state.t).total_seconds() / (60.0 * self.p.step_minutes), 0.0)
        P = state.P.copy()
        P[self.STATE, self.STATE] += self.p.q_state * steps
        P[self.DRIFT, self.DRIFT] += self.p.q_drift * steps
        return FusionState(state.x.copy(), P, until)

    def update(
        self, state: FusionState, value: float, variance: float, observes_drift: bool
    ) -> FusionState:
        """Fold in one measurement; infinite variance leaves the state alone."""
        if value is None or not np.isfinite(value) or value <= 0:
            return state
        if not np.isfinite(variance):
            return state
        H = np.array([1.0, 1.0 if observes_drift else 0.0])
        innovation = np.log(max(float(value), LOG_FLOOR)) - H @ state.x
        S = float(H @ state.P @ H + variance)
        if S <= 0:
            return state
        K = (state.P @ H) / S
        x = state.x + K * innovation
        P = (np.eye(2) - np.outer(K, H)) @ state.P
        return FusionState(x, 0.5 * (P + P.T), state.t)

    def pak_variance(self, health: str) -> float:
        """A failed or suspect analyser is admitted with infinite variance."""
        return self.p.r_pak if health == "OK" else float("inf")

    def run(self, observations: Iterable[dict]) -> list[dict]:
        """Filter a time-ordered stream of observations.

        Each observation is ``{t, source, value, health}`` where ``t`` is the
        moment the measurement refers to.  The caller is responsible for
        ordering by that moment, which is what makes a delayed laboratory
        result land correctly (see :func:`replay_with_lab_delay`).
        """
        state: Optional[FusionState] = None
        out = []
        for obs in observations:
            t = pd.Timestamp(obs["t"])
            if state is None:
                if obs["value"] is None or not np.isfinite(obs["value"]) or obs["value"] <= 0:
                    continue
                state = self.initial(obs["value"], t)
            else:
                state = self.predict(state, t)
                if obs["source"] == "LIMS":
                    state = self.update(state, obs["value"], self.p.r_lims, observes_drift=False)
                else:
                    state = self.update(
                        state,
                        obs["value"],
                        self.pak_variance(obs.get("health", "OK")),
                        observes_drift=True,
                    )
            out.append(
                {
                    "t": t,
                    "source": obs["source"],
                    "log_sulfur": float(state.x[self.STATE]),
                    "sulfur": float(np.exp(state.x[self.STATE])),
                    "drift": float(state.x[self.DRIFT]),
                    "var_state": float(state.P[self.STATE, self.STATE]),
                    "var_drift": float(state.P[self.DRIFT, self.DRIFT]),
                }
            )
        return out


def build_observations(
    lims: pd.Series,
    pak: pd.Series,
    t: pd.Timestamp,
    lab_delay_minutes: float,
    pak_health: str = "OK",
    lookback_hours: float = 240.0,
    pak_every_minutes: float = 60.0,
) -> list[dict]:
    """Observations available at ``t``, ordered by the moment they refer to.

    Two different times matter for a laboratory result: when the sample was
    taken and when the number came back.  Availability is decided by the
    second - an analysis reported after ``t`` cannot be used - while its
    position in the filter is decided by the first.  Sorting by sample time
    after filtering by readiness is what replays a delayed analysis from the
    moment it belongs to, instead of pasting a hours-old value onto now.
    """
    t = pd.Timestamp(t)
    start = t - pd.Timedelta(hours=lookback_hours)
    ready_by = t - pd.Timedelta(minutes=float(lab_delay_minutes))

    rows: list[dict] = []
    if lims is not None and len(lims):
        window = lims.loc[start:ready_by]
        rows += [
            {"t": stamp, "source": "LIMS", "value": float(value), "health": "OK"}
            for stamp, value in window.items()
            if np.isfinite(value) and value > 0
        ]
    if pak is not None and len(pak):
        window = pak.loc[start:t]
        if pak_every_minutes:
            window = window.resample(f"{int(pak_every_minutes)}min").last().dropna()
        rows += [
            {"t": stamp, "source": "PAK", "value": float(value), "health": pak_health}
            for stamp, value in window.items()
            if np.isfinite(value) and value > 0
        ]

    rows.sort(key=lambda r: (r["t"], 0 if r["source"] == "PAK" else 1))
    return rows


def nowcast_at(
    lims: pd.Series,
    pak: pd.Series,
    t: pd.Timestamp,
    lab_delay_minutes: float,
    pak_health: str = "OK",
    params: Optional[FusionParams] = None,
    **kwargs,
) -> Optional[dict]:
    """Fused estimate of sulfur as of ``t``, or None when nothing is known."""
    fusion = NowcastFusion(params)
    observations = build_observations(lims, pak, t, lab_delay_minutes, pak_health, **kwargs)
    if not observations:
        return None
    history = fusion.run(observations)
    if not history:
        return None
    last = history[-1]
    state = FusionState(
        np.array([last["log_sulfur"], last["drift"]]),
        np.diag([last["var_state"], last["var_drift"]]),
        pd.Timestamp(last["t"]),
    )
    state = fusion.predict(state, t)
    variance = float(state.P[NowcastFusion.STATE, NowcastFusion.STATE])
    return {
        "t": t,
        "sulfur": float(np.exp(state.x[NowcastFusion.STATE])),
        "log_sulfur": float(state.x[NowcastFusion.STATE]),
        "variance": variance,
        "sigma_log": float(np.sqrt(variance)),
        "drift": float(state.x[NowcastFusion.DRIFT]),
        "drift_variance": float(state.P[NowcastFusion.DRIFT, NowcastFusion.DRIFT]),
        "n_observations": len(observations),
        "last_observation": str(last["t"]),
        "last_source": last["source"],
    }
