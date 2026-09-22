"""Quality features aligned by availability time, with measurement age preserved."""

from __future__ import annotations

import pandas as pd

from src.config import load_config
from src.data.validation import causal_run_length


def pak_health_series(pak: pd.Series, cfg=None) -> pd.DataFrame:
    cfg = cfg or load_config()
    dq = cfg.main["data_quality"]
    run = causal_run_length(pak, float(dq["flatline_tolerance"]))
    suspect = run >= int(dq["flatline_min_points"])
    failed = run >= int(dq["flatline_critical_points"])
    health = pd.Series("OK", index=pak.index)
    health[suspect], health[failed] = "SUSPECT", "FAILED"
    return pd.DataFrame({"value": pak, "health": health, "suspect": suspect, "failed": failed})


def available_feature(base, series, name, delay=0, max_age=None):
    """Join on result availability; timestamps and ages still refer to sampling."""
    left = pd.DataFrame({"timestamp": pd.DatetimeIndex(base)})
    right = pd.DataFrame(
        {
            "available": series.index + pd.Timedelta(minutes=delay),
            name: series.to_numpy(),
            f"{name}_ts": series.index,
        }
    )
    out = pd.merge_asof(
        left.sort_values("timestamp"),
        right.sort_values("available"),
        left_on="timestamp",
        right_on="available",
        direction="backward",
    )
    out[f"{name}_age_min"] = (out.timestamp - out[f"{name}_ts"]).dt.total_seconds() / 60
    if max_age is not None:
        out[name] = out[name].mask(out[f"{name}_age_min"] > max_age)
    return out.drop(columns="available").set_index("timestamp")


def quality_source_features(
    timestamps,
    lims_target,
    pak_sulfur,
    lims_other=None,
    cfg=None,
    pak_health=None,
):
    """Apply the same positive LIMS delay during fitting and online inference."""
    cfg = cfg or load_config()
    dq = cfg.main["data_quality"]
    health = pak_health_series(pak_sulfur, cfg) if pak_health is None else pak_health
    healthy = pak_sulfur.mask(health["suspect"])
    frames = [
        available_feature(timestamps, healthy, "pak_sulfur", max_age=dq["pak_max_age_minutes"]),
        available_feature(
            timestamps, healthy, "pak_sulfur_healthy", max_age=dq["pak_max_age_minutes"]
        ),
        available_feature(timestamps, health["failed"].astype(float), "pak_failed_flag"),
        available_feature(timestamps, health["suspect"].astype(float), "pak_suspect_flag"),
    ]
    valid = lims_target.where(
        lims_target.between(0, cfg.main["quality"]["max_plausible_sulfur_mgkg"], inclusive="right")
    )
    frames.append(
        available_feature(
            timestamps,
            valid,
            "lims_sulfur_prev",
            dq["lims_delay_minutes"],
            dq["lims_max_age_minutes"],
        )
    )
    for name, series in (lims_other or {}).items():
        frames.append(
            available_feature(
                timestamps,
                series,
                f"lims_{name}",
                dq["lims_delay_minutes"],
                dq["feed_lims_max_age_minutes"],
            )
        )
    out = pd.concat(frames, axis=1)
    out["lims_staleness_ratio"] = out["lims_sulfur_prev_age_min"] / dq["lims_max_age_minutes"]
    out["source_disagreement_abs"] = (out.lims_sulfur_prev - out.pak_sulfur).abs()
    denom = out[["lims_sulfur_prev", "pak_sulfur"]].abs().max(axis=1).replace(0, float("nan"))
    out["source_disagreement_rel"] = out.source_disagreement_abs / denom
    return out
