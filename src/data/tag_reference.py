"""Parser and consistency checker for 'Теги_хакатон.xlsx'.

Two facts drive this module:

1. The physical meaning of a short tag (T6, F15, P24 ...) must come from the
   reference sheet, not from its first letter.
2. For the 24-2000 sheet the declared descriptions and the short tags are, in
   several rows, mutually inconsistent with the actual recorded values (see
   reports/tag_semantics.md).  The module therefore keeps BOTH the declared
   description and an empirical consistency verdict, and never silently picks
   one of them.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import List

import pandas as pd

from src.config import load_config

LETTER_CLASS = {
    "T": "temperature",
    "P": "pressure",
    "F": "flow",
    "W": "mass_flow_or_other",
    "D": "density",
    "L": "level",
    "Q": "flow_or_quality",
}

DESCRIPTION_CLASS_PATTERNS = [
    ("temperature", r"температур"),
    ("pressure", r"давлен|вакуум|перепад"),
    ("flow", r"расход|переток|производительн"),
    ("density", r"плотност"),
    ("level", r"уровен"),
    ("quality", r"анализатор|качеств|сер[ыу]|вспышк"),
]


@dataclass
class TagInfo:
    unit_id: str
    tag: str
    description: str
    letter_class: str
    description_class: str
    classes_agree: bool
    source_file: str
    source_sheet: str


def _describe_class(text: str) -> str:
    low = str(text).lower()
    for name, pattern in DESCRIPTION_CLASS_PATTERNS:
        if re.search(pattern, low):
            return name
    return "unknown"


def load_tag_reference() -> pd.DataFrame:
    cfg = load_config()
    path = cfg.raw_file("tags_update")
    rows = []
    for sheet, unit_id in (("АВТ", "AVT"), ("24-2000", "24-2000")):
        for tag, desc, unit in pd.read_excel(path, sheet_name=sheet).itertuples(
            index=False, name=None
        ):
            letter = LETTER_CLASS.get(str(tag)[:1], "unknown")
            dclass = _describe_class(str(desc))
            rows.append(
                {
                    **asdict(
                        TagInfo(
                            unit_id,
                            tag,
                            desc,
                            letter,
                            dclass,
                            _class_compatible(letter, dclass),
                            path.name,
                            sheet,
                        )
                    ),
                    "unit": unit,
                }
            )
    return pd.DataFrame(rows)


def load_legacy_tag_reference() -> pd.DataFrame:
    cfg = load_config()
    path = cfg.raw_file("tags")
    kip = pd.read_excel(path, sheet_name="КИП", header=0)
    kip.columns = [str(c).strip() for c in kip.columns]
    rows: List[TagInfo] = []
    pairs = [("АВТ (описание)", "АВТ", "AVT"), ("24-2000 (описание)", "24-2000", "24-2000")]
    for desc_col, tag_col, unit_id in pairs:
        if desc_col not in kip.columns or tag_col not in kip.columns:
            continue
        for _, r in kip[[desc_col, tag_col]].dropna().iterrows():
            tag = str(r[tag_col]).strip()
            desc = str(r[desc_col]).strip()
            letter = LETTER_CLASS.get(tag[:1].upper(), "unknown")
            dclass = _describe_class(desc)
            agree = _class_compatible(letter, dclass)
            rows.append(TagInfo(unit_id, tag, desc, letter, dclass, agree, path.name, "КИП"))
    return pd.DataFrame([asdict(r) for r in rows])


def _class_compatible(letter_class: str, description_class: str) -> bool:
    if description_class == "unknown" or letter_class == "unknown":
        return False
    if letter_class == description_class:
        return True
    if letter_class == "mass_flow_or_other" and description_class in ("flow", "pressure"):
        return True
    return bool(letter_class == "flow_or_quality" and description_class in ("flow", "quality"))


def load_pak_reference() -> pd.DataFrame:
    cfg = load_config()
    raw = pd.read_excel(cfg.raw_file("tags"), sheet_name="ПАК", header=None)
    out = []
    for _, r in raw.iterrows():
        vals = [v for v in r.tolist() if isinstance(v, str) and v.strip()]
        if len(vals) == 2:
            out.append({"description": vals[0].strip(), "raw_tag": vals[1].strip()})
    return pd.DataFrame(out)


def load_vak_reference() -> pd.DataFrame:
    """Virtual analyser formulas (sheet 'ВАК')."""
    cfg = load_config()
    path = cfg.raw_file("vak_update")
    out = []
    for sheet, frame in pd.read_excel(path, sheet_name=None).items():
        for row in frame.itertuples(index=False, name=None):
            out.append({"block": sheet, "raw_name": row[1], "raw_formula": row[2]})
    return pd.DataFrame(out)


def load_legacy_vak_reference() -> pd.DataFrame:
    cfg = load_config()
    raw = pd.read_excel(cfg.raw_file("tags"), sheet_name="ВАК", header=None)
    blocks, out = [], []
    header = raw.iloc[0].tolist()
    for col in range(0, raw.shape[1], 2):
        block = header[col] if col < len(header) and isinstance(header[col], str) else None
        blocks.append((col, block))
    for col, block in blocks:
        for i in range(1, raw.shape[0]):
            name = raw.iat[i, col] if col < raw.shape[1] else None
            expr = raw.iat[i, col + 1] if col + 1 < raw.shape[1] else None
            if isinstance(name, str) and isinstance(expr, str) and name.strip():
                out.append({"block": block, "raw_name": name.strip(), "raw_formula": expr.strip()})
    return pd.DataFrame(out)


def load_la_reference() -> pd.DataFrame:
    """Laboratory points sheet 'ЛА' (point -> list of measured properties)."""
    cfg = load_config()
    raw = pd.read_excel(cfg.raw_file("tags"), sheet_name="ЛА", header=0)
    out = []
    for col in raw.columns:
        for v in raw[col].dropna().tolist():
            out.append({"point_header": str(col).strip(), "property_ru": str(v).strip()})
    return pd.DataFrame(out)
