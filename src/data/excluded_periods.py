"""Generate dated, causal exclusion records without changing raw measurements."""

from __future__ import annotations

import pandas as pd

from src.data.operating_mode import operating_modes
from src.features.quality_features import pak_health_series


def _segments(mask: pd.Series, reason: str, detector: str, scope: str) -> list[dict]:
    groups = mask.ne(mask.shift(fill_value=False)).cumsum()
    rows = []
    for _, block in mask[mask].groupby(groups[mask]):
        rows.append(
            {
                "start": str(block.index[0]),
                "end": str(block.index[-1]),
                "reason": reason,
                "detector": detector,
                "n_points": len(block),
                "scope": scope,
            }
        )
    return rows


def exclusion_report(tel, target, pak, cfg) -> pd.DataFrame:
    modes = operating_modes(tel, cfg)
    rows = []
    for mode in ("SHUTDOWN", "TRANSIENT", "SENSOR_FAULT"):
        reason = "SENSOR_FAULT:ht:P3" if mode == "SENSOR_FAULT" else mode
        rows.extend(_segments(modes["mode"].eq(mode), reason, "causal_mode_v1", "fit_and_advice"))
    rows.extend(
        _segments(
            pak_health_series(pak, cfg)["suspect"],
            "PAK_FLATLINE",
            "causal_run_length",
            "pak_measurement",
        )
    )
    bad = ~target.between(0, cfg.main["quality"]["max_plausible_sulfur_mgkg"], inclusive="right")
    rows.extend(_segments(bad, "LIMS_IMPLAUSIBLE", "product_sulfur_range", "product_lims"))
    frame = pd.DataFrame(
        rows, columns=["start", "end", "reason", "detector", "n_points", "scope"]
    ).sort_values(["start", "reason"])
    frame["formulation"] = [
        f"исключены данные с {row.start} по {row.end} по причине: "
        f"{REASON_RU.get(row.reason, row.reason)}"
        for row in frame.itertuples()
    ]
    return frame


REASON_RU = {
    "SHUTDOWN": "установка остановлена (режим вне рабочего)",
    "TRANSIENT": "переходный режим после пуска",
    "SENSOR_FAULT:ht:P3": "отказ датчика давления ht:P3",
    "PAK_FLATLINE": "зависание поточного анализатора (флэтлайн)",
    "LIMS_IMPLAUSIBLE": "неправдоподобный лабораторный результат",
}
