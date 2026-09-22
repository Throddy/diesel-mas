"""Decide, from data, whether ht:Q20 can serve as a feed-sulfur source.

The updated tag reference calls Q20 an online sulfur analyser and its readings
sit at feed level, which makes it a candidate to replace a once-a-day
laboratory result with a ten-minute signal.  Before wiring it in, the claim has
to survive the data: an analyser that reads in the right range but does not
follow the laboratory would replace a rare accurate signal with a frequent
uninformative one.
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
from src.data.validation import find_flatlines

TAG = "ht:Q20"


def paired(
    series: pd.Series, lab: pd.Series, shift_hours: float = 0.0, tolerance: str = "6h"
) -> pd.DataFrame:
    """Align an analyser series to laboratory samples, looking backwards only."""
    shifted = series.shift(freq=pd.Timedelta(hours=shift_hours)) if shift_hours else series
    left = pd.DataFrame({"t": lab.index, "lab": lab.to_numpy()}).sort_values("t")
    right = pd.DataFrame({"t": shifted.index, "analyser": shifted.to_numpy()}).sort_values("t")
    return pd.merge_asof(
        left, right, on="t", direction="backward", tolerance=pd.Timedelta(tolerance)
    ).dropna()


def agreement(frame: pd.DataFrame) -> dict:
    """How closely the analyser tracks the laboratory on paired samples."""
    if len(frame) < 20:
        return {"n": int(len(frame)), "note": "пар недостаточно для вывода"}
    return {
        "n": int(len(frame)),
        "pearson": float(np.corrcoef(frame.lab, frame.analyser)[0, 1]),
        "spearman": float(frame.lab.corr(frame.analyser, method="spearman")),
        "median_ratio": float((frame.analyser / frame.lab).median()),
        "median_bias_mgkg": float((frame.analyser - frame.lab).median()),
        "mae_mgkg": float(np.mean(np.abs(frame.analyser - frame.lab))),
    }


def health(raw: pd.Series, cleaned: pd.Series, eligible: pd.Series, cfg) -> dict:
    """Sentinel share, flatlines and coverage of the analyser."""
    dq = cfg.main["data_quality"]
    sentinels = {
        str(value): int((raw == float(value)).sum())
        for value in cfg.main["telemetry"]["sentinel_candidates"]
    }
    running = cleaned[eligible]
    segments = find_flatlines(
        running.dropna(), int(dq["flatline_min_points"]), float(dq["flatline_tolerance"])
    )
    long_segments = [s for s in segments if s.n_points >= int(dq["flatline_critical_points"])]
    flat_points = int(sum(s.n_points for s in long_segments))
    return {
        "points_total": int(len(raw)),
        "sentinel_counts": sentinels,
        "sentinel_share": float(sum(sentinels.values()) / max(len(raw), 1)),
        "missing_after_masking_share": float(cleaned.isna().mean()),
        "running_points": int(len(running)),
        "flatline_segments_ge_24h": len(long_segments),
        "flatline_points_ge_24h": flat_points,
        "flatline_share_of_running": float(flat_points / max(len(running.dropna()), 1)),
    }


def main(tolerance: str = "6h") -> dict:
    cfg = load_config()
    interim = project_root() / cfg.main["paths"]["interim"]
    raw = load_df(interim / "telemetry.pkl")[TAG]
    tel = clean_telemetry(load_df(interim / "telemetry.pkl"), cfg)
    lims = load_df(interim / "lims_long.pkl")
    eligible = operating_modes(tel, cfg)["eligible"]

    analyser = tel[TAG][eligible].dropna().sort_index()
    feed_lab = feed_sulfur(lims).sort_index()
    product_lab = target_series(lims, cfg).sort_index()
    smooth = analyser.rolling("12h").median()

    report = {
        "tag": TAG,
        "claim": "поточный анализатор серы; по диапазону соответствует сырью, а не продукту",
        "levels": {
            "analyser_median_mgkg": float(analyser.median()),
            "feed_lab_median_mgkg": float(feed_lab.median()),
            "product_lab_median_mgkg": float(product_lab.median()),
            "analyser_over_product": float(analyser.median() / product_lab.median()),
            "analyser_p05_p95": [float(analyser.quantile(0.05)), float(analyser.quantile(0.95))],
            "feed_lab_p05_p95": [float(feed_lab.quantile(0.05)), float(feed_lab.quantile(0.95))],
        },
        "agreement_with_feed": agreement(paired(analyser, feed_lab, tolerance=tolerance)),
        "agreement_with_product": agreement(paired(analyser, product_lab, tolerance=tolerance)),
        "agreement_by_shift": {
            f"{shift:+d}h": agreement(paired(smooth, feed_lab, shift, tolerance)).get("pearson")
            for shift in (-48, -24, -12, -6, 0, 6, 12, 24)
        },
        "agreement_by_year": {
            str(year): agreement(
                paired(smooth, feed_lab[feed_lab.index.year == year], tolerance=tolerance)
            ).get("pearson")
            for year in sorted(feed_lab.index.year.unique())
        },
        "variability": {
            "feed_lab_cv": float(feed_lab.std() / feed_lab.mean()),
            "analyser_cv_12h": float(smooth.std() / smooth.mean()),
        },
        "health": health(raw, tel[TAG], eligible, cfg),
    }

    level_ok = report["levels"]["analyser_over_product"] > 100
    tracks = (report["agreement_with_feed"].get("pearson") or 0) >= 0.5
    stable = all((v or 0) >= 0.3 for v in report["agreement_by_year"].values() if v is not None)
    report["verdict"] = {
        "reads_at_feed_level": bool(level_ok),
        "tracks_laboratory_feed_sulfur": bool(tracks),
        "relationship_stable_over_years": bool(stable),
        "usable_as_feed_sulfur_source": bool(level_ok and tracks and stable),
        "rule": (
            "подключать только если анализатор читает на уровне сырья И следует за "
            "лабораторной серой сырья (Пирсон не ниже 0,5) И связь устойчива по годам"
        ),
    }
    out = project_root() / cfg.main["paths"]["reports"] / "q20_assessment.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "levels": report["levels"],
                "agreement_with_feed": report["agreement_with_feed"],
                "agreement_by_year": report["agreement_by_year"],
                "verdict": report["verdict"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\n-> {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="проверка назначения ht:Q20 по данным")
    ap.add_argument("--tolerance", default="6h")
    main(**vars(ap.parse_args()))
