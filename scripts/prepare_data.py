"""Stage 1: raw package -> canonical layer + control support ranges + audit."""

from __future__ import annotations

import argparse
import json

from src.config import load_config, project_root
from src.data.loaders import (
    build_canonical,
    clean_telemetry,
    pak_d15,
    pak_sulfur,
    sentinel_report,
    split_bounds,
    target_series,
)
from src.data.store import save_df
from src.data.tag_reference import load_tag_reference
from src.data.validation import (
    find_flatlines,
    gap_report,
    inter_arrival_stats,
    saturation_hypothesis,
)


def _longest(flat):
    if not flat:
        return None
    seg = max(flat, key=lambda f: f.n_points)
    return {
        "start": str(seg.start),
        "end": str(seg.end),
        "hours": round(seg.duration_hours, 1),
        "n_points": seg.n_points,
        "value": float(seg.value),
    }


def main(force: bool = False) -> dict:
    cfg = load_config()
    out = build_canonical(cfg, force=force)
    tel, lims, pak = out["telemetry"], out["lims"], out["pak"]
    report = dict(out["report"])

    clean = clean_telemetry(tel, cfg)
    sent = sentinel_report(tel, cfg)
    save_df(sent, project_root() / cfg.main["paths"]["interim"] / "sentinels.pkl", csv_twin=True)
    report["sentinel_columns"] = int(len(sent))
    report["sentinel_top"] = sent.head(10).to_dict("records")

    train_end, valid_end, test_start = split_bounds(cfg)
    from src.data.operating_mode import operating_modes

    modes = operating_modes(clean, cfg)
    train_tel = clean.loc[:train_end].where(modes["eligible"])

    support = {}
    ql, qh = cfg.main["optimization"]["support_quantiles"]
    dq = float(cfg.main["optimization"]["delta_quantile"])
    for ctrl in cfg.controls["controls"]:
        entry = {
            "canonical_name": ctrl["canonical_name"],
            "tags": {},
            "industrial_limit_known": bool(ctrl.get("industrial_limit_known", False)),
            "note": "ASSUMPTION - NOT INDUSTRIAL LIMIT (robust TRAIN quantiles)",
        }
        for tag in ctrl["tags"]:
            if tag not in train_tel.columns:
                continue
            s = train_tel[tag].dropna()
            d = train_tel[tag].diff().abs().dropna()
            entry["tags"][tag] = {
                "low": float(s.quantile(ql)),
                "high": float(s.quantile(qh)),
                "median": float(s.median()),
                "step_q95": float(d.quantile(dq)),
                "n": int(len(s)),
            }
        support[ctrl["canonical_name"]] = entry
    (project_root() / cfg.main["paths"]["models"]).mkdir(parents=True, exist_ok=True)
    with open(
        project_root() / cfg.main["paths"]["models"] / "controls_support.json",
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(support, fh, ensure_ascii=False, indent=2)
    report["controls_supported"] = list(support)

    y = target_series(lims, cfg)
    ps, pd15s = pak_sulfur(pak), pak_d15(pak)
    dqc = cfg.main["data_quality"]
    flat = find_flatlines(ps, int(dqc["flatline_min_points"]), float(dqc["flatline_tolerance"]))
    flat_crit = [f for f in flat if f.n_points >= int(dqc["flatline_critical_points"])]
    report["lims_target"] = {
        "n": int(len(y)),
        "t_min": str(y.index.min()),
        "t_max": str(y.index.max()),
        "n_above_limit": int((y > cfg.sulfur_limit).sum()),
        "inter_arrival": inter_arrival_stats(y.index),
        "max": float(y.max()),
        "min": float(y.min()),
    }
    report["pak_sulfur"] = {
        "n": int(len(ps)),
        "t_min": str(ps.index.min()),
        "t_max": str(ps.index.max()),
        "flatline_segments_ge_6h": len(flat),
        "flatline_segments_ge_24h": len(flat_crit),
        "flatlined_share": float(sum(f.n_points for f in flat) / max(len(ps), 1)),
        "longest_flatline": _longest(flat),
        "saturation_hypothesis": saturation_hypothesis(ps),
        "gaps": gap_report(ps.index, cfg.main["telemetry"]["expected_step_minutes"]),
    }
    report["pak_d15"] = {
        "n": int(len(pd15s)),
        "t_min": str(pd15s.index.min()),
        "t_max": str(pd15s.index.max()),
    }
    report["lims_nonnumeric"] = int(lims["data_quality_flags"].str.contains("NONNUMERIC").sum())
    report["lims_duplicate_timestamps"] = int(
        lims["data_quality_flags"].str.contains("DUPLICATE").sum()
    )
    report["split"] = {
        "train_end": str(train_end),
        "valid_end": str(valid_end),
        "test_start": str(test_start),
    }

    from src.data.excluded_periods import exclusion_report

    exclusions = exclusion_report(clean, y, ps, cfg)
    exclusions.to_csv(cfg.path("reports") / "excluded_periods.csv", index=False)
    report["mode_counts"] = modes["mode"].value_counts().to_dict()
    report["excluded_counts"] = exclusions.groupby("reason")["n_points"].sum().to_dict()
    tags = load_tag_reference()
    from src.data.tag_reference import load_legacy_tag_reference

    old = load_legacy_tag_reference()
    comparison = old.merge(tags, on=["unit_id", "tag"], suffixes=("_old", "_new"))
    comparison.to_csv(cfg.path("reports") / "tag_reference_changes.csv", index=False)
    report["tag_reference"] = {
        "n": int(len(tags)),
        "class_conflicts": tags.groupby("unit_id")["classes_agree"]
        .apply(lambda s: int((~s).sum()))
        .to_dict(),
    }
    from src.models.vak import VAK_CACHE, parse_reference_formulas, save_formulas

    formulas = parse_reference_formulas()
    save_formulas(formulas, project_root() / cfg.main["paths"]["interim"] / VAK_CACHE)
    report["vak_formulas"] = {
        "n": len(formulas),
        "by_status": {
            status: sum(1 for f in formulas if f.status == status)
            for status in sorted({f.status for f in formulas})
        },
    }

    art = project_root() / cfg.main["paths"]["reports"]
    art.mkdir(parents=True, exist_ok=True)
    with open(art / "data_audit.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=str)
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("telemetry_shape", "lims_rows", "pak_rows", "lims_target", "pak_sulfur")
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild the canonical layer from scratch")
    main(**vars(ap.parse_args()))
