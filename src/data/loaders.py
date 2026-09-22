"""Single entry point that turns the raw package into the canonical data layer.

Nothing downstream ever touches the raw spreadsheets directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.config import Config, load_config, project_root
from src.data.lims_parser import lims_series, parse_lims
from src.data.pak_parser import pak_series, parse_pak
from src.data.store import load_df, save_df
from src.data.telemetry import grid_report, load_telemetry

TELEMETRY_PKL = "telemetry.pkl"
LIMS_PKL = "lims_long.pkl"
PAK_PKL = "pak_long.pkl"


def ensure_telemetry_csv(cfg: Optional[Config] = None) -> Dict[str, Path]:
    """Resolve source CSVs or extract the archive exclusively into interim/unpacked.

    The archive itself is never modified; extraction goes to data/interim/unpacked
    and raw sources are never modified.
    """
    cfg = cfg or load_config()
    raw = cfg.raw_file("archive").parent
    dest = project_root() / cfg.main["paths"]["interim"] / "unpacked"
    targets = {}
    for key in ("avt", "ht"):
        name = cfg.main["raw_files"][f"telemetry_{key}"]
        matches = list(dest.rglob(name)) if dest.exists() else []
        targets[key] = (
            raw / name if (raw / name).exists() else (matches[0] if matches else dest / name)
        )
    if all(p.exists() for p in targets.values()):
        return targets
    archive = raw / cfg.main["raw_files"]["archive"]
    if not archive.exists():
        raise FileNotFoundError(
            f"neither the telemetry CSVs nor {archive.name} were found in {raw}"
        )
    dest = project_root() / cfg.main["paths"]["interim"] / "unpacked"
    dest.mkdir(parents=True, exist_ok=True)
    _extract_archive(archive, dest)
    for key, target in targets.items():
        if target.exists():
            continue
        found = list(dest.rglob(target.name))
        if not found:
            raise FileNotFoundError(f"{target.name} not found inside {archive.name}")
        targets[key] = found[0]
    return targets


def _extract_archive(archive: Path, dest: Path) -> None:
    """Extract a .rar/.zip archive without any external binary if possible."""
    import zipfile

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        return
    try:
        from src.data.libarchive_extract import extract_with_libarchive

        extract_with_libarchive(archive, dest)
        return
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"could not extract {archive.name} automatically ({exc}). "
            f"Please unpack it manually into {dest}"
        ) from exc


def build_canonical(cfg: Optional[Config] = None, force: bool = False) -> Dict[str, object]:
    cfg = cfg or load_config()
    interim = project_root() / cfg.main["paths"]["interim"]
    interim.mkdir(parents=True, exist_ok=True)
    report: Dict[str, object] = {}

    tel_path = interim / TELEMETRY_PKL
    if force or not tel_path.exists():
        csvs = ensure_telemetry_csv(cfg)
        avt = load_telemetry(csvs["avt"], "avt")
        ht = load_telemetry(csvs["ht"], "ht")
        report["avt_grid"] = grid_report(avt, cfg.main["telemetry"]["expected_step_minutes"])
        report["ht_grid"] = grid_report(ht, cfg.main["telemetry"]["expected_step_minutes"])
        report["avt_dropped_index_columns"] = avt.attrs.get("dropped_index_columns", [])
        report["ht_dropped_index_columns"] = ht.attrs.get("dropped_index_columns", [])
        tel = avt.join(ht, how="outer").sort_index()
        save_df(tel, tel_path)
    else:
        tel = load_df(tel_path)

    lims_path = interim / LIMS_PKL
    if force or not lims_path.exists():
        lims = parse_lims(cfg.raw_file("lims"))
        save_df(lims, lims_path)
    else:
        lims = load_df(lims_path)

    pak_path = interim / PAK_PKL
    if force or not pak_path.exists():
        pak = parse_pak(cfg.raw_file("pak"))
        save_df(pak, pak_path)
    else:
        pak = load_df(pak_path)

    report["telemetry_shape"] = list(tel.shape)
    report["lims_rows"] = int(len(lims))
    report["pak_rows"] = int(len(pak))
    return {"telemetry": tel, "lims": lims, "pak": pak, "report": report}


def clean_telemetry(tel: pd.DataFrame, cfg: Optional[Config] = None) -> pd.DataFrame:
    """Mask sentinel values (see assumption A-07) and drop dead constant columns.

    The masked values are NOT dropped from the audit: scripts/prepare_data.py
    reports how many values each column loses.
    """
    cfg = cfg or load_config()
    out = tel.copy()
    from src.data.validation import causal_run_length

    simultaneous = tel.eq(307.0).sum(axis=1) >= 3
    for col in out:
        for sentinel in cfg.main["telemetry"]["sentinel_candidates"]:
            eq = out[col].eq(float(sentinel))
            long = causal_run_length(out[col]) >= 3
            out[col] = out[col].mask(eq & (simultaneous | long))
    return out


def sentinel_report(tel: pd.DataFrame, cfg: Optional[Config] = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    rows = []
    for sentinel in cfg.main["telemetry"]["sentinel_candidates"]:
        eq = tel == float(sentinel)
        counts = eq.sum()
        for col, n in counts[counts > 0].items():
            rows.append(
                {"column": col, "sentinel": sentinel, "n": int(n), "share": float(n / len(tel))}
            )
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def target_series(lims: pd.DataFrame, cfg: Optional[Config] = None) -> pd.Series:
    """Product sulfur (LIMS, hydrotreating sample point 2), mg/kg."""
    cfg = cfg or load_config()
    return lims_series(lims, "Гидроочистка", "2", cfg.main["quality"]["target_metric"])


def pak_sulfur(pak: pd.DataFrame) -> pd.Series:
    return pak_series(pak, "Mg.Sulfur")


def pak_d15(pak: pd.DataFrame) -> pd.Series:
    return pak_series(pak, "D15")


def split_bounds(cfg: Optional[Config] = None):
    cfg = cfg or load_config()
    s = cfg.main["split"]
    return (
        pd.Timestamp(s["train_end"]),
        pd.Timestamp(s["valid_end"]),
        pd.Timestamp(s["test_start"]),
    )


def product_t95(lims: pd.DataFrame) -> pd.Series:
    """Laboratory 95 % distillation point of the hydrotreated product, in °C.

    The soft-sensor formula 24-2000:GODT:T95 needs the previous laboratory
    value as one of its terms, so it is read from the same LIMS point as the
    target sulfur (point 2 of unit 24-2000).
    """
    return lims_series(lims, "Гидроочистка", "2", "95%.T").rename("product_t95_c")


def feed_sulfur(lims: pd.DataFrame) -> pd.Series:
    """Feed LIMS is Mass.Sulfur in mass percent; convert explicitly to mg/kg."""
    s = lims_series(lims, "Гидроочистка", "1", "Mass.Sulfur")
    return (s[s.between(0, 100, inclusive="right")] * 10000.0).rename("feed_sulfur_mgkg")
