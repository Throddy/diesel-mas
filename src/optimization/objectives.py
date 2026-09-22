"""Secondary objectives.

Everything here is a PROXY built only from tags whose meaning is verified.
No monetary values are invented: the package contains no economic data
(NOT CONFIRMED FROM PROVIDED DATA).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

PRODUCTION_TAG = "ht:F17"

ENERGY_TERMS = {}


def production_proxy(
    telemetry: Dict[str, Optional[float]], feed_relative_change: float = 0.0
) -> Optional[float]:
    """Product flow; a feed-rate move is propagated proportionally."""
    base = telemetry.get(PRODUCTION_TAG)
    if base is None or not np.isfinite(base):
        return None
    return float(base) * (1.0 + feed_relative_change)


def energy_proxy(
    telemetry: Dict[str, Optional[float]], medians: Dict[str, float]
) -> Optional[float]:
    """Relative heat-duty proxy F*(T-Tref), normalized by training medians."""
    from src.config import load_config

    reference = load_config().main["optimization"]["heat_reference_c"]
    feed, temp = telemetry.get("ht:F9"), telemetry.get("ht:T5")
    if feed is None or temp is None or not np.isfinite([feed, temp]).all():
        return None
    denom = medians.get("ht:F9", 0) * (medians.get("ht:T5", reference) - reference)
    return float(feed * max(0.0, temp - reference) / denom) if denom > 0 else None


def energy_formula_text() -> str:
    return (
        "Прокси тепловой нагрузки, F9 умножить на (T5 минус Tref) и поделить на "
        "значение по медианам обучающего блока. Tref задан допущением в "
        "конфигурации. Величина безразмерная, измеренной энергией не является."
    )
