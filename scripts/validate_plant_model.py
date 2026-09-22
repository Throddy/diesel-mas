"""Check the kinetic plant model against history, without fitting it.

The model is calibrated at a single operating point.  Running it over the
recorded regimes and comparing with the laboratory shows how far that one
point carries: a model that is out by a factor is not a physical model of this
unit and must not be presented as one.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.config import load_config, project_root
from src.data.loaders import clean_telemetry, feed_sulfur, target_series
from src.data.operating_mode import operating_modes
from src.data.store import load_df
from src.sim.plant_model import PlantModel, PlantParams

TAGS = ("ht:T5", "ht:F9", "ht:P3")


def operating_frame(cfg) -> tuple[pd.DataFrame, pd.Series]:
    """Laboratory analyses joined with the regime that produced them."""
    interim = project_root() / cfg.main["paths"]["interim"]
    tel = clean_telemetry(load_df(interim / "telemetry.pkl"), cfg)
    lims = load_df(interim / "lims_long.pkl")
    eligible = operating_modes(tel, cfg)["eligible"]

    product = target_series(lims, cfg)
    product = product[(product > 0) & (product <= 100)]
    feed = feed_sulfur(lims).sort_index()

    lag = float(cfg.main["response"]["lag_minutes"])
    at = product.index - pd.Timedelta(minutes=lag)
    regime = tel[list(TAGS)].reindex(at, method="ffill").set_axis(product.index)
    regime["feed_sulfur_mgkg"] = feed.reindex(at, method="ffill").to_numpy()
    running = eligible.reindex(at, method="ffill").fillna(False).to_numpy()
    keep = running & regime.notna().all(axis=1).to_numpy() & (regime > 0).all(axis=1).to_numpy()
    return regime.loc[keep], product.loc[keep]


def calibrate_on_train(cfg, regime: pd.DataFrame, product: pd.Series) -> PlantModel:
    """Pin the one free constant to the median training regime."""
    train_end = pd.Timestamp(cfg.main["split"]["train_end"])
    mask = regime.index <= train_end
    reference = regime.loc[mask].median()
    target = float(product.loc[mask].median())
    return PlantModel(PlantParams()).calibrate(
        feed_sulfur_mgkg=float(reference["feed_sulfur_mgkg"]),
        temperature_c=float(reference["ht:T5"]),
        feed_tph=float(reference["ht:F9"]),
        pressure_mpa=float(reference["ht:P3"]),
        product_sulfur_mgkg=target,
    )


def score(plant: PlantModel, regime: pd.DataFrame, product: pd.Series) -> dict:
    """Agreement between the model and the laboratory on one block."""
    predicted = np.array(
        [
            plant.steady_state(row["feed_sulfur_mgkg"], row["ht:T5"], row["ht:F9"], row["ht:P3"])
            for _, row in regime.iterrows()
        ]
    )
    truth = product.to_numpy(float)
    ratio = predicted / np.maximum(truth, 1e-9)
    return {
        "n": int(len(truth)),
        "MAE_mgkg": float(np.mean(np.abs(predicted - truth))),
        "MedianAE_mgkg": float(np.median(np.abs(predicted - truth))),
        "bias_mgkg": float(np.median(predicted - truth)),
        "median_ratio": float(np.median(ratio)),
        "ratio_p05_p95": [float(np.quantile(ratio, 0.05)), float(np.quantile(ratio, 0.95))],
        "within_factor_2": float(np.mean((ratio > 0.5) & (ratio < 2.0))),
        "correlation": float(
            np.corrcoef(np.log(np.maximum(predicted, 1e-9)), np.log(np.maximum(truth, 1e-9)))[0, 1]
        ),
        "predicted_median": float(np.median(predicted)),
        "truth_median": float(np.median(truth)),
    }


def main() -> dict:
    cfg = load_config()
    regime, product = operating_frame(cfg)
    plant = calibrate_on_train(cfg, regime, product)
    train_end = pd.Timestamp(cfg.main["split"]["train_end"])
    test_start = pd.Timestamp(cfg.main["split"]["test_start"])

    blocks = {
        "train": (regime.index <= train_end),
        "test": (regime.index >= test_start),
        "all": np.ones(len(regime), dtype=bool),
    }
    report = {"model": plant.describe(), "blocks": {}}
    for name, mask in blocks.items():
        if mask.sum() < 5:
            continue
        report["blocks"][name] = score(plant, regime.loc[mask], product.loc[mask])

    reference = regime.median()
    report["sensitivity_comparison"] = {
        "plant_d_ln_s_dt": plant.temperature_sensitivity(
            float(reference["feed_sulfur_mgkg"]),
            float(reference["ht:T5"]),
            float(reference["ht:F9"]),
            float(reference["ht:P3"]),
        ),
        "model_M_d_ln_s_dt": json.loads(
            (project_root() / cfg.main["paths"]["models"] / "response_model.json").read_text(
                encoding="utf-8"
            )
        )["temperature_log_sensitivity_joint"],
    }
    test = report["blocks"].get("test", {})
    within = test.get("within_factor_2", 0.0)
    report["verdict"] = {
        "usable_as_a_simulated_plant": bool(within >= 0.5),
        "physically_validated": False,
        "note": (
            "модель калибруется одной точкой и не обучается на данных; "
            "она пригодна как независимый отклик для замкнутого контура, "
            "но не является проверенной физической моделью установки"
        ),
    }
    out = project_root() / cfg.main["paths"]["reports"] / "plant_model_validation.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "blocks": report["blocks"],
                "sensitivity_comparison": report["sensitivity_comparison"],
                "verdict": report["verdict"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\n-> {out}")
    return report


if __name__ == "__main__":
    argparse.ArgumentParser(description="проверка кинетической модели на истории").parse_args()
    main()
