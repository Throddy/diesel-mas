"""Explicit, reversible unit registry.

Nothing is converted implicitly: a conversion exists only if it is registered
here together with the assumption that justifies it. The unit ``ppm`` is
accepted only as mass ppm, which is assumption A-03 rather than a documented
property of the analyser.
"""

from __future__ import annotations

from typing import Optional, Tuple

from src.config import load_config

CONVERSIONS = {
    ("mg_per_kg", "mg_per_kg"): (1.0, 0.0, None),
    ("ppm_mass", "mg_per_kg"): (1.0, 0.0, "A-03"),
    ("mass_pct", "mg_per_kg"): (1.0e4, 0.0, "A-04"),
    ("deg_c", "deg_c"): (1.0, 0.0, None),
    ("kg_per_m3", "kg_per_m3"): (1.0, 0.0, None),
    ("vol_pct", "vol_pct"): (1.0, 0.0, None),
    ("cetane_units", "cetane_units"): (1.0, 0.0, None),
}


def canonical_unit_token(raw_unit: Optional[str]) -> str:
    """Map a raw unit string from a file to a canonical token."""
    if raw_unit is None:
        return "unknown"
    key = str(raw_unit).strip()
    table = load_config().aliases["units"]
    if key in table:
        token = table[key]
        return "ppm_mass" if key == "ppm" else token
    return "unknown"


def convert(
    value: float, from_token: str, to_token: str
) -> Tuple[Optional[float], Optional[str], bool]:
    """Return (converted_value, assumption_id, ok).

    ok=False means no registered conversion -> the caller must keep the original
    unit and raise a data-quality warning instead of guessing.
    """
    if value is None:
        return None, None, False
    key = (from_token, to_token)
    if key not in CONVERSIONS:
        return None, None, False
    factor, offset, assumption = CONVERSIONS[key]
    return value * factor + offset, assumption, True
