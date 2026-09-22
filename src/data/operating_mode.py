"""Causal operating-mode classification with explicit modelling thresholds."""

from __future__ import annotations

import pandas as pd

from src.config import load_config


def operating_modes(tel: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """Use only past transitions; never retrospectively mark pre-shutdown rows."""
    cfg = cfg or load_config()
    c = cfg.main["operating_mode"]
    temp, feed, pressure = (tel[c[k]] for k in ("temperature_tag", "feed_tag", "pressure_tag"))
    hot = temp >= c["temperature_min_c"]
    flowing = feed >= c["feed_min_tph"]
    shutdown = (temp < c["temperature_min_c"]) & (feed < c["feed_min_tph"])
    fault = hot & flowing & ((pressure < c["pressure_min_mpa"]) | pressure.isna())
    ready = hot & flowing & (pressure >= c["pressure_min_mpa"])
    mode = pd.Series("TRANSIENT", index=tel.index)
    mode.loc[shutdown] = "SHUTDOWN"
    changed = ready.ne(ready.shift(fill_value=False))
    changes = pd.Series(tel.index.where(changed), index=tel.index).ffill()
    age = (pd.Series(tel.index, index=tel.index) - changes).dt.total_seconds() / 60
    mode.loc[ready & (age >= c["transient_minutes"])] = "RUNNING"
    mode.loc[fault] = "SENSOR_FAULT"
    return pd.DataFrame({"mode": mode, "pressure_fault": fault, "eligible": mode.eq("RUNNING")})
