"""Catalyst activity index as a feature.

The index is the temperature the response model M would need to reproduce the
sulfur the laboratory actually found.  A rising index means the same severity
buys less desulfurisation, which is what deactivation looks like from outside.
The 30-day version reported in `scripts/catalyst_activity.py` is too noisy to
predict with; over 90 days it describes the phase of the run.

Two rules keep the feature honest.  M is fitted on the training block only, so
a feature used on validation and test is never built from their labels.  The
index at moment `t` is read as of `t` minus the laboratory delay, so the
analysis the row is labelled with cannot enter its own feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.operating_mode import operating_modes
from src.features.quality_features import available_feature
from src.models.response_model import TAGS, ResponseModel, design


def model_log_sulfur(response: ResponseModel, frame: pd.DataFrame) -> np.ndarray:
    """ln S predicted by M at the given operating point."""
    z = (design(frame) - response.center) / response.scale
    return response.coefficient[0] + z @ response.coefficient[1:]


def operating_points(
    tel: pd.DataFrame, target: pd.Series, feed: pd.Series, cfg
) -> tuple[pd.DataFrame, pd.Series]:
    """Levers and feed sulfur at each analysis, restricted to a normal mode."""
    dq = cfg.main["data_quality"]
    lag = pd.Timedelta(minutes=float(cfg.main["response"]["lag_minutes"]))
    y = target[
        target.between(0, cfg.main["quality"]["max_plausible_sulfur_mgkg"], inclusive="right")
    ]
    at = y.index - lag
    frame = tel[list(TAGS[:3])].reindex(at, method="ffill").set_axis(y.index)
    ready = available_feature(
        at, feed, "feed_sulfur_mgkg", dq["lims_delay_minutes"], dq["feed_lims_max_age_minutes"]
    )
    frame["feed_sulfur_mgkg"] = ready["feed_sulfur_mgkg"].to_numpy()
    eligible = operating_modes(tel, cfg)["eligible"].reindex(at, method="ffill")
    keep = (
        eligible.fillna(False).to_numpy()
        & frame.notna().all(axis=1).to_numpy()
        & (frame > 0).all(axis=1).to_numpy()
    )
    return frame.loc[keep], y.loc[keep]


def index_series(
    tel: pd.DataFrame, target: pd.Series, feed: pd.Series, response: ResponseModel, cfg
) -> pd.Series:
    """Required temperature shift at every usable laboratory analysis."""
    frame, y = operating_points(tel, target, feed, cfg)
    if frame.empty:
        return pd.Series(dtype=float, name="required_temperature_shift_c")
    predicted = model_log_sulfur(response, frame)
    kelvin = frame["ht:T5"].to_numpy(float) + 273.15
    sensitivity = (response.coefficient[1] / response.scale[0]) * (-1000.0 / kelvin**2)
    shift = (np.log(y.to_numpy(float)) - predicted) / sensitivity
    return pd.Series(shift, index=y.index, name="required_temperature_shift_c")


def fit_reference_model(
    tel: pd.DataFrame, target: pd.Series, feed: pd.Series, cfg, fit_end
) -> ResponseModel:
    """Fit M for feature use on the training block alone."""
    end = pd.Timestamp(fit_end)
    return ResponseModel.fit(
        tel.loc[:end], target[target.index <= end], feed[feed.index <= end], cfg
    )


class CatalystIndexFeature:
    """The index as one as-of feature, computed once and read at any moment.

    Building it fits M and inverts it at every analysis, which is far too slow
    to repeat inside a decision cycle, so the whole series is prepared once and
    each moment is a lookup.
    """

    def __init__(
        self,
        tel: pd.DataFrame,
        target: pd.Series,
        feed: pd.Series,
        cfg,
        window_days: int = 90,
        response: ResponseModel | None = None,
    ):
        self.name = f"catalyst_index_median{window_days}d"
        if response is None:
            response = fit_reference_model(tel, target, feed, cfg, cfg.main["split"]["train_end"])
        raw = index_series(tel, target, feed, response, cfg)
        delay = pd.Timedelta(minutes=float(cfg.main["data_quality"]["lims_delay_minutes"]))
        smooth = raw.rolling(f"{window_days}D").median() if not raw.empty else raw
        self.known = smooth.copy()
        if not self.known.empty:
            self.known.index = self.known.index + delay

    def at(self, moments: pd.DatetimeIndex) -> pd.DataFrame:
        moments = pd.DatetimeIndex(moments)
        if self.known.empty:
            return pd.DataFrame({self.name: np.nan}, index=moments)
        merged = self.known[~self.known.index.duplicated(keep="last")]
        values = merged.reindex(merged.index.union(moments)).ffill().reindex(moments)
        return pd.DataFrame({self.name: values.to_numpy(float)}, index=moments)


def index_feature(
    tel: pd.DataFrame,
    target: pd.Series,
    feed: pd.Series,
    moments: pd.DatetimeIndex,
    cfg,
    window_days: int = 90,
    response: ResponseModel | None = None,
) -> pd.DataFrame:
    """One-shot form of :class:`CatalystIndexFeature`."""
    return CatalystIndexFeature(tel, target, feed, cfg, window_days, response).at(moments)
