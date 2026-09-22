"""LIMS parser.

Layout discovered in the file (verified, not assumed):
  row 0  - group header  "Установка 'X'. Точка отбора 'Y'. Продукт 'Z'" (merged,
           only the first column of the group carries the text)
  row 1  - metric name              (per timestamp/value pair)
  row 2  - unit string              (per pair)
  row 3  - "Количество значений:" / declared count
  row 4+ - alternating timestamp / value columns

The file is therefore 54 independent, asynchronous series side by side.  The
parser returns a long Observation table; nothing is aligned by row number.
"""

from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.aliases import canonical_unit, resolve_metric
from src.data.units import canonical_unit_token, convert

GROUP_RE = re.compile(r"Установка\s*'(?P<unit>[^']+)'\.*\s*Точка отбора\s*'(?P<point>[^']+)'")


def _parse_group(text: str) -> Dict[str, str]:
    m = GROUP_RE.search(str(text))
    if not m:
        return {"process_unit": "UNKNOWN", "sample_point": str(text)[:60]}
    unit = m.group("unit").strip()
    unit = "Гидроочистка" if unit.lower().startswith("гидроочист") else unit
    return {"process_unit": unit, "sample_point": m.group("point").strip()}


def parse_lims(path=None) -> pd.DataFrame:
    cfg = load_config()
    path = path or cfg.raw_file("lims")
    raw = pd.read_excel(path, sheet_name=0, header=None)
    records: List[dict] = []
    group = {"process_unit": "UNKNOWN", "sample_point": "UNKNOWN"}
    for c in range(0, raw.shape[1] - 1, 2):
        header = raw.iat[0, c]
        if isinstance(header, str) and header.strip():
            group = _parse_group(header)
        raw_metric = raw.iat[1, c]
        raw_unit = raw.iat[2, c]
        declared_count = raw.iat[3, c + 1]
        if not isinstance(raw_metric, str) or not raw_metric.strip():
            continue
        canonical = resolve_metric(raw_metric) or raw_metric.strip()
        from_token = canonical_unit_token(raw_unit)
        to_token = (
            canonical_unit_token(canonical_unit(canonical))
            if canonical_unit(canonical)
            else from_token
        )
        ts_col = raw.iloc[4:, c]
        val_col = raw.iloc[4:, c + 1]
        ts = pd.to_datetime(ts_col, errors="coerce")
        numeric = pd.to_numeric(val_col, errors="coerce")
        nonnumeric_mask = numeric.isna() & val_col.notna()
        for i in range(len(ts)):
            t = ts.iloc[i]
            if pd.isna(t):
                continue
            flags: List[str] = []
            v = numeric.iloc[i]
            if nonnumeric_mask.iloc[i]:
                flags.append(f"NONNUMERIC_VALUE:{str(val_col.iloc[i])[:24]}")
                v = np.nan
            if pd.isna(v) and not flags:
                continue
            value, assumption, ok = (
                (float(v), None, True) if pd.isna(v) else convert(float(v), from_token, to_token)
            )
            if not ok:
                flags.append(f"UNIT_NOT_CONVERTIBLE:{from_token}->{to_token}")
                value = float(v)
                canonical_out_unit = raw_unit
            else:
                canonical_out_unit = canonical_unit(canonical) or raw_unit
                if assumption:
                    flags.append(f"UNIT_ASSUMPTION:{assumption}")
            records.append(
                {
                    "timestamp": t,
                    "process_unit": group["process_unit"],
                    "sample_point": group["sample_point"],
                    "canonical_metric": canonical,
                    "raw_metric": str(raw_metric).strip(),
                    "value": value,
                    "original_unit": str(raw_unit),
                    "canonical_unit": canonical_out_unit,
                    "source": "LIMS",
                    "data_quality_flags": ";".join(flags),
                    "original_reference": f"{path.name}#col{c}",
                    "declared_count": float(declared_count) if pd.notna(declared_count) else np.nan,
                }
            )
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    df["series_id"] = df["process_unit"] + "|" + df["sample_point"] + "|" + df["canonical_metric"]
    dup = df.duplicated(subset=["series_id", "timestamp"], keep=False)
    df.loc[dup, "data_quality_flags"] = (
        df.loc[dup, "data_quality_flags"] + ";DUPLICATE_TIMESTAMP"
    ).str.strip(";")
    return df.sort_values(["series_id", "timestamp"]).reset_index(drop=True)


def lims_series(df: pd.DataFrame, process_unit: str, sample_point: str, metric: str) -> pd.Series:
    sel = df[
        (df.process_unit == process_unit)
        & (df.sample_point == sample_point)
        & (df.canonical_metric == metric)
        & df.value.notna()
    ]
    s = sel.set_index("timestamp")["value"].sort_index()
    return s[~s.index.duplicated(keep="last")]
