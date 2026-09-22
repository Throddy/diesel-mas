"""Supervised dataset assembly for the product-sulfur model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.alignment import assert_no_future
from src.features.controller_activity import activity_frame
from src.features.process_features import build_trailing_features, selected_tags
from src.features.quality_features import quality_source_features


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame
    feature_names: List[str]
    lag_minutes: int
    horizon_minutes: float = 0.0

    def slice(self, start=None, end=None) -> "Dataset":
        m = pd.Series(True, index=self.X.index)
        if start is not None:
            m &= self.X.index >= pd.Timestamp(start)
        if end is not None:
            m &= self.X.index <= pd.Timestamp(end)
        return Dataset(
            self.X[m.to_numpy()],
            self.y[m.to_numpy()],
            self.meta[m.to_numpy()],
            self.feature_names,
            self.lag_minutes,
            self.horizon_minutes,
        )


def build_dataset(
    tel: pd.DataFrame,
    target: pd.Series,
    pak_sulfur: pd.Series,
    lims_other: Optional[Dict[str, pd.Series]] = None,
    lag_minutes: int = 0,
    cfg=None,
    horizon_minutes: float = 0.0,
) -> Dataset:
    """Assemble features as of the decision moment and targets at their own time.

    ``horizon_minutes`` moves the decision moment back from the target, so a
    horizon model sees only what was available that many minutes earlier.
    """
    cfg = cfg or load_config()
    qcfg = cfg.main["quality"]
    windows = qcfg["trailing_windows_minutes"]

    y = target.copy()
    y = y[np.isfinite(y.to_numpy())]
    keep = (y > 0) & (y <= float(qcfg["max_plausible_sulfur_mgkg"]))
    dropped = int((~keep).sum())
    y = y[keep]
    from src.data.operating_mode import operating_modes

    modes = operating_modes(tel, cfg)
    eligibility = modes["eligible"].reindex(y.index, method="ffill").fillna(False)
    y = y[eligibility & (y.index <= tel.index.max())]

    horizon = pd.Timedelta(minutes=float(horizon_minutes))
    decision = pd.DatetimeIndex(y.index) - horizon

    tags = selected_tags(tel)
    proc, feat_ts = build_trailing_features(tel, tags, windows, decision, lag_minutes=lag_minutes)
    decided = pd.Series(decision, index=y.index)
    assert_no_future(pd.Series(feat_ts.to_numpy(), index=y.index), decided)

    qual = quality_source_features(decision, target, pak_sulfur, lims_other, cfg)
    stamp_columns = [c for c in qual.columns if c.endswith("_ts")]
    for column in stamp_columns:
        values = pd.Series(qual[column].to_numpy(), index=y.index)
        known = values.notna()
        if known.any():
            assert_no_future(values[known], decided[known])
    result_delay = pd.Timedelta(minutes=float(cfg.main["data_quality"]["lims_delay_minutes"]))
    meta = pd.DataFrame(
        {
            "feature_timestamp": feat_ts.to_numpy(),
            "decision_time": decision,
            "sample_time": pd.DatetimeIndex(y.index),
            "result_available_time": pd.DatetimeIndex(y.index) + result_delay,
        },
        index=y.index,
    )
    for column in stamp_columns:
        meta[column] = qual[column].to_numpy()
    qual = qual.drop(columns=stamp_columns)

    X = pd.concat([proc.reset_index(drop=True), qual.reset_index(drop=True)], axis=1)
    X.index = y.index
    blocks = _optional_blocks(tel, target, lims_other, decision, lag_minutes, cfg)
    X = pd.concat([X] + [block.set_axis(y.index) for block in blocks], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).astype("float32")
    X = X.loc[:, X.notna().any()]
    meta["n_dropped_implausible_targets"] = dropped
    return Dataset(
        X=X,
        y=y,
        meta=meta,
        feature_names=list(X.columns),
        lag_minutes=lag_minutes,
        horizon_minutes=float(horizon_minutes),
    )


def chronological_split(ds: Dataset, cfg=None) -> Dict[str, Dataset]:
    cfg = cfg or load_config()
    s = cfg.main["split"]
    train_end, valid_end, test_start = (
        pd.Timestamp(s["train_end"]),
        pd.Timestamp(s["valid_end"]),
        pd.Timestamp(s["test_start"]),
    )
    return {
        "train": ds.slice(end=train_end),
        "valid": ds.slice(start=train_end + pd.Timedelta(seconds=1), end=valid_end),
        "test": ds.slice(start=test_start),
    }


@dataclass
class TrainingBlocks:
    """Disjoint chronological blocks used to fit, select and calibrate."""

    train: Dataset
    selection: Dataset
    calibration: Dataset
    training_cutoff: pd.Timestamp
    calibration_cutoff: pd.Timestamp
    mode: str


def available_until(timestamps, cutoff, delay_minutes: float) -> np.ndarray:
    """Mask of analyses whose result is reported no later than ``cutoff``."""
    reported = pd.DatetimeIndex(timestamps) + pd.Timedelta(minutes=float(delay_minutes))
    return np.asarray(reported <= pd.Timestamp(cutoff))


def training_blocks(ds: Dataset, cfg=None, mode: Optional[str] = None) -> TrainingBlocks:
    """Split a dataset into fitting, selection and calibration blocks.

    ``fixed`` keeps the chronological split declared in the configuration.
    ``expanding`` uses every analysis whose result is already reported, holds
    out the most recent ones for calibration and the ones before them for model
    selection, so the fitting sample grows with the available history.
    """
    cfg = cfg or load_config()
    mode = mode or cfg.main.get("retraining", {}).get("mode", "fixed")
    split = cfg.main["split"]
    if mode == "fixed":
        train_end = pd.Timestamp(split["train_end"])
        calibration_start = pd.Timestamp(cfg.main["quality"]["calibration_start"])
        valid_end = pd.Timestamp(split["valid_end"]) + pd.Timedelta(days=1)
        return TrainingBlocks(
            train=ds.slice(end=train_end),
            selection=ds.slice(
                start=train_end + pd.Timedelta(seconds=1),
                end=calibration_start - pd.Timedelta(seconds=1),
            ),
            calibration=ds.slice(start=calibration_start, end=valid_end),
            training_cutoff=train_end,
            calibration_cutoff=valid_end,
            mode=mode,
        )

    retraining = cfg.main["retraining"]
    holdout = int(retraining["calibration_holdout"])
    selection_size = int(retraining.get("selection_holdout", holdout))
    total = len(ds.y)
    if total < holdout + selection_size + holdout:
        raise ValueError(f"недостаточно анализов для режима expanding: {total}")
    calibration_from = total - holdout
    selection_from = calibration_from - selection_size
    index = pd.DatetimeIndex(ds.y.index)
    return TrainingBlocks(
        train=ds.slice(end=index[selection_from - 1]),
        selection=ds.slice(start=index[selection_from], end=index[calibration_from - 1]),
        calibration=ds.slice(start=index[calibration_from]),
        training_cutoff=index[selection_from - 1],
        calibration_cutoff=index[-1],
        mode=mode,
    )


def _optional_blocks(tel, target, lims_other, moments, lag_minutes, cfg) -> List[pd.DataFrame]:
    """Feature blocks added after the first submission, each behind a flag.

    Both were measured on their own before being switched on; the numbers are
    in `reports/feature_blocks.json`.  They stay optional so a run can
    reproduce the submitted configuration by turning them off.
    """
    flags = cfg.main.get("features", {})
    blocks: List[pd.DataFrame] = []
    if flags.get("controller_activity", False):
        blocks.append(activity_frame(tel, moments, lag_minutes=lag_minutes))
    if flags.get("catalyst_index", False):
        from src.features.catalyst_index import index_feature

        feed = (lims_other or {}).get("feed_sulfur")
        if feed is not None:
            blocks.append(
                index_feature(
                    tel,
                    target,
                    feed,
                    moments,
                    cfg,
                    int(flags.get("catalyst_index_window_days", 90)),
                )
            )
    return blocks


def build_dataset_for_model(
    tel: pd.DataFrame,
    lims: pd.DataFrame,
    pak: pd.DataFrame,
    model,
    cfg=None,
    horizon_minutes: float = 0.0,
) -> Dataset:
    """Build the dataset a fitted model expects, with the lag it was fitted at.

    Every evaluation script used to assemble the features itself, and each one
    could - and did - forget a group the training run had included, failing
    with "Missing model features" (K-09).  Going through one function keeps
    training and every downstream use in step.
    """
    from src.data.loaders import feed_sulfur, pak_sulfur, target_series

    cfg = cfg or load_config()
    return build_dataset(
        tel,
        target_series(lims, cfg),
        pak_sulfur(pak),
        {"feed_sulfur": feed_sulfur(lims)},
        lag_minutes=model.lag_minutes,
        cfg=cfg,
        horizon_minutes=horizon_minutes,
    )
