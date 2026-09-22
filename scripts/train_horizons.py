"""Fit one sulfur model per forecast horizon on disjoint chronological blocks."""

from __future__ import annotations

import argparse
import json

from src.config import load_config, project_root
from src.data.loaders import clean_telemetry, feed_sulfur, pak_sulfur, target_series
from src.data.store import load_df
from src.models.horizons import build_horizon_dataset, fit_horizon, horizon_hours


def main() -> dict:
    cfg = load_config()
    interim, models = cfg.path("interim"), cfg.path("models")
    telemetry = clean_telemetry(load_df(interim / "telemetry.pkl"), cfg)
    lims, pak = load_df(interim / "lims_long.pkl"), load_df(interim / "pak_long.pkl")
    target, feed = target_series(lims, cfg), feed_sulfur(lims)

    trained = []
    for hours in horizon_hours(cfg):
        dataset = build_horizon_dataset(
            telemetry, target, pak_sulfur(pak), {"feed_sulfur": feed}, hours, cfg
        )
        horizon = fit_horizon(dataset, hours, cfg)
        artifact = horizon.save(models)
        trained.append(
            {
                "horizon_hours": hours,
                "n_targets": int(len(dataset.y)),
                "n_train": horizon.n_train,
                "n_calibration": horizon.n_calibration,
                "training_cutoff": horizon.training_cutoff,
                "calibration_cutoff": horizon.calibration_cutoff,
                "status": horizon.status,
                "artifact": f"models/{artifact.name}",
            }
        )
        print(json.dumps(trained[-1], ensure_ascii=False), flush=True)

    report = {
        "horizons": trained,
        "definition": "горизонт h означает прогноз на t_decision плюс h часов; "
        "признаки собираются на t_decision",
    }
    out = project_root() / cfg.main["paths"]["reports"] / "horizons_training.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")
    return report


if __name__ == "__main__":
    argparse.ArgumentParser(description="обучение моделей по горизонтам").parse_args()
    main()
