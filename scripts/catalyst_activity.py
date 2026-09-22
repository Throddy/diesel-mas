"""Catalyst activity index: the temperature the model would need today.

For every laboratory analysis the response model M is inverted: given the
measured feed rate and feed sulfur, which reactor temperature would reproduce
the sulfur the laboratory actually found?  The gap between that required
temperature and the temperature the plant actually ran at is the index.  A
rising gap means the same severity buys less desulfurisation, which is what
catalyst deactivation looks like from outside.

This is an indicator, not a measurement: there are no regeneration dates in
the package, so nothing here is calendar age of the catalyst (A-10).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.config import load_config, project_root
from src.data.loaders import clean_telemetry, feed_sulfur, target_series
from src.data.store import load_df
from src.features.catalyst_index import index_series
from src.models.response_model import ResponseModel


def main(window_days: int = 30) -> dict:
    cfg = load_config()
    interim = project_root() / cfg.main["paths"]["interim"]
    models = project_root() / cfg.main["paths"]["models"]
    response = ResponseModel.load(models / "response_model.joblib")
    tel = clean_telemetry(load_df(interim / "telemetry.pkl"), cfg)
    lims = load_df(interim / "lims_long.pkl")

    index = index_series(tel, target_series(lims, cfg), feed_sulfur(lims), response, cfg)
    if index.empty:
        raise SystemExit("нет пригодных анализов для расчёта индекса")

    recent = index.loc[index.index.max() - pd.Timedelta(days=window_days) :]
    trend = None
    if len(recent) >= 5:
        days = (recent.index - recent.index[0]).days.to_numpy(float)
        slope, _ = np.polyfit(days, recent.to_numpy(float), 1)
        trend = float(slope * 30.0)

    report = {
        "definition": (
            "требуемая температура минус фактическая: на сколько градусов пришлось бы "
            "изменить T5, чтобы модель M воспроизвела фактическую серу при текущих "
            "нагрузке и сере сырья"
        ),
        "interpretation": "рост индекса означает, что та же жёсткость режима даёт меньшую очистку",
        "caveat": (
            "индикатор, а не измерение: дат регенерации в пакете нет, календарный возраст "
            "катализатора не считается (A-10)"
        ),
        "n_analyses": int(len(index)),
        "period": [str(index.index.min()), str(index.index.max())],
        "index_median_c": float(np.median(index)),
        "index_last_c": float(index.iloc[-1]),
        "index_mean_last_window_c": float(recent.mean()) if len(recent) else None,
        "window_days": window_days,
        "trend_c_per_month": trend,
        "series_tail": [
            {"timestamp": str(t), "required_shift_c": round(float(v), 3)}
            for t, v in index.tail(20).items()
        ],
    }
    out = project_root() / cfg.main["paths"]["reports"] / "catalyst_activity.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "n_analyses",
                    "period",
                    "index_median_c",
                    "index_last_c",
                    "index_mean_last_window_c",
                    "trend_c_per_month",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\n-> {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="индекс активности катализатора")
    ap.add_argument("--window-days", type=int, default=30)
    main(**{k.replace("-", "_"): v for k, v in vars(ap.parse_args()).items()})
