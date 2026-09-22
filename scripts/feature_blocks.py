"""Measure what each new feature block adds, one block at a time.

Two blocks are tested.  Controller activity turns the movement the
natural-experiment search called noise into a signal: range, reversals, travel
and slope of the three levers over 6, 12, 24 and 72 hours.  The catalyst index
over a 90-day median describes the phase of the run, where the 30-day version
reported by `scripts.catalyst_activity` is too noisy to predict with.

Each block is scored against the same baseline, so the contributions do not
mix, and the pair is scored together to show whether they overlap.  The choice
of which blocks to keep is made on validation; the test block is opened once,
at the end, after that choice is already fixed.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.config import load_config, project_root
from src.data.loaders import clean_telemetry
from src.data.store import load_df
from src.features.dataset import build_dataset_for_model, chronological_split
from src.models.quality import SulfurModel, calibrate_model, make_model, regression_metrics

BLOCKS = {
    "базовый": {"controller_activity": False, "catalyst_index": False},
    "активность регулятора": {"controller_activity": True, "catalyst_index": False},
    "индекс катализатора": {"controller_activity": False, "catalyst_index": True},
    "оба блока": {"controller_activity": True, "catalyst_index": True},
}


def score(split, cfg, open_test: bool) -> dict:
    """Fit on the training block, report validation, then optionally the test."""
    train, valid, test = split["train"], split["valid"], split["test"]
    model = SulfurModel("rf", 0, [], make_model("rf", cfg.seed), limit=cfg.sulfur_limit)
    model.fit(train.X, train.y)
    out = {
        "n_features": int(train.X.shape[1]),
        "validation": regression_metrics(valid.y.to_numpy(), model.predict(valid.X)),
    }
    if open_test:
        calibrate_model(model, valid.X, valid.y, cfg)
        point, low, high, _ = model.predict_with_interval(test.X)
        truth = test.y.to_numpy()
        over = truth > cfg.sulfur_limit
        flagged = high > cfg.sulfur_limit
        out["test"] = {
            **regression_metrics(truth, point),
            "coverage": float(np.mean((truth >= low) & (truth <= high))),
            "mean_width": float(np.mean(high - low)),
            "recall_over_limit": float(np.mean(flagged[over])) if over.any() else None,
            "precision_over_limit": float(np.mean(over[flagged])) if flagged.any() else None,
        }
    return out


def main(window_days: int = 90) -> dict:
    cfg = load_config()
    interim = project_root() / cfg.main["paths"]["interim"]
    reference = SulfurModel.load(
        project_root() / cfg.main["paths"]["models"] / "sulfur_model.joblib"
    )
    tel = clean_telemetry(load_df(interim / "telemetry.pkl"), cfg)
    lims, pak = load_df(interim / "lims_long.pkl"), load_df(interim / "pak_long.pkl")

    report = {
        "catalyst_index_window_days": window_days,
        "selection": "состав блоков выбран по валидации, тест открыт один раз",
        "blocks": {},
    }
    for name, flags in BLOCKS.items():
        cfg.main["features"].update(flags)
        cfg.main["features"]["catalyst_index_window_days"] = window_days
        dataset = build_dataset_for_model(tel, lims, pak, reference, cfg)
        report["blocks"][name] = score(chronological_split(dataset, cfg), cfg, open_test=True)

    base = report["blocks"]["базовый"]
    for row in report["blocks"].values():
        row["validation"]["delta_MAE_vs_base"] = float(
            row["validation"]["MAE"] - base["validation"]["MAE"]
        )
        row["test"]["delta_MAE_vs_base"] = float(row["test"]["MAE"] - base["test"]["MAE"])
    out = project_root() / cfg.main["paths"]["reports"] / "feature_blocks.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    head = (
        f"{'блок':24s} {'призн.':>6s} {'вал.MAE':>8s} {'вал.д':>7s} "
        f"{'тест.MAE':>8s} {'тест.д':>7s} {'покр.':>6s} {'ширина':>7s}"
    )
    print(head)
    for name, row in report["blocks"].items():
        t, v = row["test"], row["validation"]
        print(
            f"{name:24s} {row['n_features']:6d} {v['MAE']:8.4f} {v['delta_MAE_vs_base']:+7.4f} "
            f"{t['MAE']:8.4f} {t['delta_MAE_vs_base']:+7.4f} "
            f"{t['coverage']:6.3f} {t['mean_width']:7.2f}"
        )
    print(f"\n-> {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="вклад отдельных блоков признаков")
    ap.add_argument("--window-days", type=int, default=90)
    main(**{k.replace("-", "_"): v for k, v in vars(ap.parse_args()).items()})
