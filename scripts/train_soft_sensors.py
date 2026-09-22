"""Inventory the laboratory properties and fit a soft sensor for each eligible one."""

from __future__ import annotations

import argparse
import json

from src.config import load_config, project_root
from src.data.loaders import clean_telemetry
from src.data.store import load_df
from src.models import soft_sensors
from src.models.uncertainty import STATUS_OK

ARTIFACT = "soft_sensors.joblib"


def main() -> dict:
    cfg = load_config()
    interim, models = cfg.path("interim"), cfg.path("models")
    telemetry = clean_telemetry(load_df(interim / "telemetry.pkl"), cfg)
    lims = load_df(interim / "lims_long.pkl")

    fitted = []
    inventory = []
    for target in soft_sensors.declared_targets(cfg):
        sensor = soft_sensors.fit(telemetry, lims, target, cfg)
        fitted.append(sensor)
        inventory.append(
            {
                "name": target.name,
                "metric": target.metric,
                "process_unit": target.process_unit,
                "sample_point": target.sample_point,
                "unit": target.unit,
                "source": "ЛИМС",
                "result_delay_minutes": float(cfg.main["data_quality"]["lims_delay_minutes"]),
                "limit": target.limit,
                "limit_source": target.limit_source or "не подтверждён источником",
                "min_observations": int(cfg.main["soft_sensors"]["min_observations"]),
                "status": sensor.status,
                **sensor.metrics,
            }
        )
        print(
            json.dumps(
                {k: inventory[-1][k] for k in ("name", "unit", "n_observations", "status")},
                ensure_ascii=False,
            ),
            flush=True,
        )

    soft_sensors.save(fitted, models / ARTIFACT)
    report = {
        "artifact": f"models/{ARTIFACT}",
        "rule": "софт-сенсор используется, если он обходит базовую оценку по "
        "предыдущему анализу и калибровочная выборка достаточна",
        "usable": [row["name"] for row in inventory if row["status"] == STATUS_OK],
        "targets": inventory,
    }
    out = project_root() / cfg.main["paths"]["reports"] / "soft_sensors.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"\n{'показатель':26s} {'ед.':>6s} {'набл.':>6s} {'сравн.':>6s} {'MAE':>7s} "
        f"{'база':>7s} {'MedAE':>7s} {'RMSE':>7s} {'R2':>6s} {'смещ.':>7s} "
        f"{'покр.':>6s} {'ширина':>7s} статус"
    )
    for row in inventory:
        print(
            f"{row['name'].split('|')[-1]:26s} {row['unit']:>6s} "
            f"{row['n_observations']:6d} {row.get('n_compared', 0):6d} "
            f"{_num(row.get('MAE')):>7s} {_num(row.get('baseline_previous_value_MAE')):>7s} "
            f"{_num(row.get('MedianAE')):>7s} {_num(row.get('RMSE')):>7s} "
            f"{_num(row.get('R2')):>6s} {_num(row.get('bias')):>7s} "
            f"{_num(row.get('interval_coverage')):>6s} "
            f"{_num(row.get('interval_mean_width'), 2):>7s} {row['status']}"
        )
    print(f"\n-> {out}")
    return report


def _num(value, digits: int = 3) -> str:
    return "нет" if value is None else f"{value:.{digits}f}"


if __name__ == "__main__":
    argparse.ArgumentParser(description="обучение софт-сенсоров ВАК").parse_args()
    main()
