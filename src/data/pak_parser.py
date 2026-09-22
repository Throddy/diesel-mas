"""PAK (online analyser) parser.

Layout: repeating blocks of [timestamp, value] columns; row 0 holds the raw tag
name, row 1 the unit, data starts at row 2.  The exported tag name
('24-2000:Mg.Sulfur') differs from the reference name ('24-2000:Mg.Sulfur.Q'),
so resolution goes through the canonical alias registry.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from src.config import load_config
from src.data.aliases import canonical_unit, resolve_metric
from src.data.units import canonical_unit_token, convert


def parse_pak(path=None) -> pd.DataFrame:
    cfg = load_config()
    path = path or cfg.raw_file("pak")
    raw = pd.read_excel(path, sheet_name=0, header=None)
    records: List[dict] = []
    for c in range(0, raw.shape[1]):
        tag = raw.iat[0, c]
        if not isinstance(tag, str) or not tag.strip():
            continue
        unit_raw = raw.iat[1, c]
        ts = pd.to_datetime(raw.iloc[2:, c], errors="coerce")
        vals = pd.to_numeric(raw.iloc[2:, c + 1], errors="coerce") if c + 1 < raw.shape[1] else None
        if vals is None:
            continue
        canonical = resolve_metric(tag) or tag.strip()
        from_token = canonical_unit_token(unit_raw)
        target_unit = canonical_unit(canonical)
        to_token = canonical_unit_token(target_unit) if target_unit else from_token
        ok_mask = ts.notna() & vals.notna()
        sub_ts = ts[ok_mask]
        sub_v = vals[ok_mask].astype(float)
        conv, assumption, ok = convert(1.0, from_token, to_token)
        flags = ""
        if ok:
            sub_v = sub_v * conv
            unit_out = target_unit or str(unit_raw)
            if assumption:
                flags = f"UNIT_ASSUMPTION:{assumption}"
        else:
            unit_out = str(unit_raw)
            flags = f"UNIT_NOT_CONVERTIBLE:{from_token}->{to_token}"
        block = pd.DataFrame(
            {
                "timestamp": sub_ts.values,
                "value": sub_v.values,
            }
        )
        block["process_unit"] = "Гидроочистка"
        block["sample_point"] = "PAK"
        block["canonical_metric"] = canonical
        block["raw_metric"] = tag.strip()
        block["original_unit"] = str(unit_raw)
        block["canonical_unit"] = unit_out
        block["source"] = "PAK"
        block["data_quality_flags"] = flags
        block["original_reference"] = f"{path.name}#col{c}"
        records.append(block)
    if not records:
        return pd.DataFrame()
    df = pd.concat(records, ignore_index=True)
    df["series_id"] = df["process_unit"] + "|PAK|" + df["canonical_metric"]
    return df.sort_values(["series_id", "timestamp"]).reset_index(drop=True)


def pak_series(df: pd.DataFrame, metric: str) -> pd.Series:
    sel = df[(df.canonical_metric == metric) & df.value.notna()]
    s = sel.set_index("timestamp")["value"].sort_index()
    return s[~s.index.duplicated(keep="last")]
