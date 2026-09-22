"""Fit the sulfur model, choose it chronologically and calibrate it separately."""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List

import pandas as pd

from src.config import load_config
from src.data.loaders import clean_telemetry, feed_sulfur, pak_sulfur, target_series
from src.data.operating_mode import operating_modes
from src.data.store import load_df, save_df
from src.features.dataset import build_dataset, training_blocks
from src.models.quality import (
    SulfurModel,
    calibrate_model,
    make_model,
    permutation_importance_df,
    regression_metrics,
)
from src.models.reliability import ReliabilityModel
from src.models.response_model import ResponseModel

ROLLING_ORIGIN_FRACTIONS = (0.6, 0.8)


def source_cutoff(cfg, mode: str) -> pd.Timestamp:
    """Latest sampling moment whose laboratory result is reported by the cutoff."""
    if mode == "expanding":
        return pd.Timestamp.max
    valid_end = pd.Timestamp(cfg.main["split"]["valid_end"]) + pd.Timedelta(days=1)
    delay = pd.Timedelta(minutes=float(cfg.main["data_quality"]["lims_delay_minutes"]))
    return valid_end - delay


def rolling_origin_metrics(kind: str, train, columns: List[str], cfg) -> List[dict]:
    """Chronological folds inside the fitting block, never on later blocks."""
    folds = []
    for fraction in ROLLING_ORIGIN_FRACTIONS:
        cut = int(len(train.y) * fraction)
        stop = min(len(train.y), cut + max(20, len(train.y) // 5))
        fold = SulfurModel(kind, 0, [], make_model(kind, cfg.seed))
        fold.fit(train.X[columns].iloc[:cut], train.y.iloc[:cut])
        folds.append(
            regression_metrics(
                train.y.iloc[cut:stop].to_numpy(), fold.predict(train.X[columns].iloc[cut:stop])
            )
        )
    return folds


def main() -> Dict[str, object]:
    started = time.monotonic()
    cfg = load_config()
    interim, models = cfg.path("interim"), cfg.path("models")
    mode = cfg.main.get("retraining", {}).get("mode", "fixed")
    cutoff = source_cutoff(cfg, mode)
    if mode == "expanding":
        print(
            json.dumps(
                {
                    "retraining_mode": mode,
                    "warning": "блок оценки входит в обучение поздних окон; его метрики "
                    "не измеряют обобщение, сравнение схем в "
                    "reports/walk_forward.json",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    telemetry = clean_telemetry(load_df(interim / "telemetry.pkl"), cfg).loc[:cutoff]
    lims = load_df(interim / "lims_long.pkl")
    lims = lims[lims.timestamp <= cutoff]
    pak = load_df(interim / "pak_long.pkl")
    pak = pak[pak.timestamp <= cutoff]

    target, feed = target_series(lims, cfg), feed_sulfur(lims)
    dataset = build_dataset(telemetry, target, pak_sulfur(pak), {"feed_sulfur": feed}, cfg=cfg)
    blocks = training_blocks(dataset, cfg, mode)
    columns = blocks.train.X.columns[blocks.train.X.notna().any()].tolist()

    selection: List[dict] = []
    fitted: Dict[str, SulfurModel] = {}
    for kind in cfg.main["quality"]["model_kinds"]:
        model = SulfurModel(
            kind,
            0,
            [],
            make_model(kind, cfg.seed),
            limit=cfg.sulfur_limit,
            model_version=cfg.model_version,
        )
        model.fit(blocks.train.X[columns], blocks.train.y)
        row = {
            "kind": kind,
            "lag_minutes": 0,
            **regression_metrics(
                blocks.selection.y.to_numpy(), model.predict(blocks.selection.X[columns])
            ),
            "rolling_origin": rolling_origin_metrics(kind, blocks.train, columns, cfg),
        }
        selection.append(row)
        fitted[kind] = model
        print(json.dumps(row, ensure_ascii=False), flush=True)

    chosen = min(selection, key=lambda row: row["MAE_log"])
    model = fitted[chosen["kind"]]
    calibrate_model(model, blocks.calibration.X[columns], blocks.calibration.y, cfg)
    model.metrics = {
        "selected": chosen,
        "retraining_mode": mode,
        "training_cutoff": str(blocks.training_cutoff),
        "calibration_cutoff": str(blocks.calibration_cutoff),
        "train_median_sulfur": float(blocks.train.y.median()),
        "train_size": len(blocks.train.y),
        "selection_size": len(blocks.selection.y),
        "calibration_size": len(blocks.calibration.y),
        "feature_count": len(columns),
        "conformal": {
            "alpha": model.calibration.alpha,
            "n_calibration": model.calibration.n,
            "status": model.calibration.status,
            "q_log": model.calibration.quantile(),
        },
    }
    model.save(models / "sulfur_model.joblib")

    importance = permutation_importance_df(
        model, blocks.selection.X[columns], blocks.selection.y, n_repeats=1, seed=cfg.seed
    )
    save_df(importance, models / "feature_importance.pkl", csv_twin=True)

    running = operating_modes(telemetry, cfg)["eligible"]
    fit_end = blocks.training_cutoff
    reliability = ReliabilityModel.fit(
        telemetry.loc[:fit_end][running.loc[:fit_end]], cfg, cfg.seed
    )
    reliability.save(models / "reliability_model.joblib")

    response = ResponseModel.fit(telemetry, target, feed, cfg)
    response.save(models / "response_model.joblib")
    (models / "response_model.json").write_text(
        json.dumps(response.diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (models / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    card = {
        "model_version": cfg.model_version,
        "selected": chosen,
        **model.metrics,
        "response": response.diagnostics,
        "train_seconds": time.monotonic() - started,
    }
    (models / "model_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(card, ensure_ascii=False, indent=2))
    return card


if __name__ == "__main__":
    argparse.ArgumentParser(description="обучение модели прогноза серы").parse_args()
    main()
