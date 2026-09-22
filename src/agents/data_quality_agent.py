"""DataQualityAgent: builds the online ProcessState and judges whether the
inputs may be trusted at all.

Everything is evaluated with backward-only information: at decision time ``t``
only measurements with ``timestamp <= t`` exist for this agent.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.alignment import asof_value
from src.data.validation import find_flatlines, robust_z, staleness_ratio
from src.schemas import DataQualityAssessment, ProcessState, SourceStatus


class DataQualityAgent:
    name = "DataQualityAgent"

    def __init__(
        self,
        telemetry: pd.DataFrame,
        lims_target: pd.Series,
        pak_sulfur: pd.Series,
        pak_d15: Optional[pd.Series] = None,
        lims_long: Optional[pd.DataFrame] = None,
        cfg=None,
        feed_sulfur: Optional[pd.Series] = None,
    ):
        self.cfg = cfg or load_config()
        self.tel = telemetry
        self.lims = lims_target
        self.pak = pak_sulfur
        self.pak_d15 = pak_d15
        self.feed_sulfur = feed_sulfur
        self.lims_long = lims_long
        self._lims_stats = {"q95_min": self.cfg.dq("lims_max_age_minutes")}
        from src.data.operating_mode import operating_modes

        self.modes = operating_modes(telemetry, self.cfg)
        self._global_flags = self._global_lims_flags()

    def _global_lims_flags(self) -> Dict[str, int]:
        if self.lims_long is None:
            return {"nonnumeric": 0, "duplicate_timestamps": 0}
        f = self.lims_long["data_quality_flags"].fillna("")
        return {
            "nonnumeric": int(f.str.contains("NONNUMERIC").sum()),
            "duplicate_timestamps": int(f.str.contains("DUPLICATE").sum()),
        }

    def _pak_health(self, t: pd.Timestamp) -> Dict[str, object]:
        """Analyzer health from the PAST window only."""
        dq = self.cfg.main["data_quality"]
        crit = int(dq["flatline_critical_points"])
        window = self.pak.loc[:t].tail(crit * 2)
        if window.empty:
            return {
                "health": "MISSING",
                "flatline_points": 0,
                "flat_value": None,
                "flat_hours": 0.0,
            }
        segs = find_flatlines(
            window, int(dq["flatline_min_points"]), float(dq["flatline_tolerance"])
        )
        trailing = [s for s in segs if s.end == window.index[-1]]
        if not trailing:
            return {"health": "OK", "flatline_points": 0, "flat_value": None, "flat_hours": 0.0}
        seg = trailing[-1]
        health = "FAILED" if seg.n_points >= crit else "SUSPECT"
        return {
            "health": health,
            "flatline_points": int(seg.n_points),
            "flat_value": float(seg.value),
            "flat_hours": round(seg.duration_hours, 1),
        }

    def run(self, t: pd.Timestamp) -> ProcessState:
        cfg, dq = self.cfg, self.cfg.main["data_quality"]
        t = pd.Timestamp(t)
        flags: List[str] = []
        notes: List[str] = []

        tel_hist = self.tel.loc[:t]
        if tel_hist.empty:
            raise ValueError(f"no telemetry at or before {t}")
        row = tel_hist.iloc[-1].copy()
        mode = self.modes.loc[:t].iloc[-1]["mode"]
        if mode != "RUNNING":
            flags.append(f"MODE:{mode}")
        if mode == "SENSOR_FAULT":
            flags.append("SENSOR_FAULT:ht:P3")
            row["ht:P3"] = np.nan
        tel_ts = tel_hist.index[-1]
        tel_age = (t - tel_ts).total_seconds() / 60.0
        missing_share = float(row.isna().mean())
        if tel_age > float(dq["telemetry_max_age_minutes"]):
            flags.append(f"TELEMETRY_STALE:{tel_age:.0f}min")
        if missing_share > 0.25:
            flags.append(f"TELEMETRY_MISSING:{missing_share:.0%}")

        stale_tags: List[str] = []
        recent = tel_hist.tail(int(dq["low_variance_window"]))
        if len(recent) >= int(dq["low_variance_window"]):
            stds = recent.std(ddof=0)
            stale_tags = [
                c
                for c in stds.index
                if np.isfinite(stds[c]) and stds[c] < float(dq["low_variance_min_std"])
            ]
            if stale_tags:
                flags.append(f"STUCK_SIGNALS:{len(stale_tags)}")

        for ctrl in cfg.controls["controls"]:
            for tag in ctrl["tags"]:
                if tag not in tel_hist.columns:
                    continue
                s = tel_hist[tag].tail(144).dropna()
                if len(s) < 20:
                    continue
                z = robust_z(s.diff().dropna())
                if np.isfinite(z.iloc[-1]) and abs(z.iloc[-1]) > float(dq["jump_robust_z"]):
                    flags.append(f"SUDDEN_JUMP:{tag}")

        lims_v, lims_ts, lims_age = asof_value(
            self.lims, t - pd.Timedelta(minutes=dq["lims_delay_minutes"])
        )
        if lims_age is not None:
            lims_age += dq["lims_delay_minutes"]
        ratio = staleness_ratio(lims_age, self._lims_stats)
        lims_health = "MISSING" if lims_v is None else "OK"
        if ratio is not None and ratio > 1.0:
            lims_health = "SUSPECT"
            flags.append(f"LIMS_STALE:ratio={ratio:.2f}")
        if (
            lims_v is not None
            and not 0 < lims_v <= cfg.main["quality"]["max_plausible_sulfur_mgkg"]
        ):
            lims_health = "IMPLAUSIBLE"
            flags.append("LIMS_IMPLAUSIBLE")
        lims_status = SourceStatus(
            source="LIMS",
            metric=cfg.main["quality"]["target_metric"],
            last_timestamp=lims_ts,
            value=lims_v,
            unit=cfg.main["quality"]["target_unit"],
            age_minutes=lims_age,
            staleness_ratio=ratio,
            health=lims_health,
            flags=[
                f"global_nonnumeric={self._global_flags['nonnumeric']}",
                f"global_duplicate_ts={self._global_flags['duplicate_timestamps']}",
            ],
        )

        pak_v, pak_ts, pak_age = asof_value(self.pak, t)
        health_info = self._pak_health(t)
        pak_health = health_info["health"]
        if pak_v is None:
            pak_health = "MISSING"
        elif pak_age is not None and pak_age > float(dq["pak_max_age_minutes"]):
            pak_health = "SUSPECT"
            flags.append(f"PAK_STALE:{pak_age:.0f}min")
        if pak_health in ("SUSPECT", "FAILED"):
            flags.append(f"PAK_FLATLINE:{health_info['flat_hours']}h@{health_info['flat_value']}")
        pak_status = SourceStatus(
            source="PAK",
            metric="Mg.Sulfur",
            last_timestamp=pak_ts,
            value=pak_v,
            unit=cfg.main["quality"]["target_unit"],
            age_minutes=pak_age,
            staleness_ratio=None,
            health=pak_health,
            flags=[
                f"flatline_points={health_info['flatline_points']}",
                "unit ppm interpreted as mg/kg (assumption A-03)",
            ],
        )

        sources = [lims_status, pak_status]
        if self.pak_d15 is not None and len(self.pak_d15):
            d15_v, d15_ts, d15_age = asof_value(self.pak_d15, t)
            d15_health = (
                "MISSING"
                if d15_v is None
                else ("SUSPECT" if d15_age and d15_age > 24 * 60 else "OK")
            )
            if d15_v is None:
                notes.append(
                    "PAK D15 has no coverage before 2025-03-05 (source coverage differs by signal)"
                )
            sources.append(
                SourceStatus("PAK", "D15", d15_ts, d15_v, "кг/м3", d15_age, None, d15_health, [])
            )

        if self.feed_sulfur is not None and len(self.feed_sulfur):
            ready = self.feed_sulfur.loc[
                : t - pd.Timedelta(minutes=float(dq["lims_delay_minutes"]))
            ]
            fs_v, fs_ts, fs_age = asof_value(ready, t)
            max_age = float(dq["feed_lims_max_age_minutes"])
            fs_health = (
                "MISSING"
                if fs_v is None
                else "SUSPECT" if fs_age is not None and fs_age > max_age else "OK"
            )
            if fs_v is None:
                notes.append(
                    "feed sulfur (LIMS 24-2000 point 1) has no analysis before this moment"
                )
            elif fs_health == "SUSPECT":
                flags.append(f"FEED_SULFUR_STALE:{fs_age:.0f}min")
            sources.append(
                SourceStatus(
                    "LIMS",
                    "Feed.Sulfur",
                    fs_ts,
                    fs_v,
                    cfg.main["quality"]["target_unit"],
                    fs_age,
                    None,
                    fs_health,
                    [
                        "точка 1 ЛИМС 24-2000: сера сырья гидроочистки",
                        f"доступна с модельной задержкой {dq['lims_delay_minutes']} мин",
                    ],
                )
            )

        disagreement = None
        if lims_v is not None and pak_v is not None:
            diff = abs(lims_v - pak_v)
            rel = diff / max(abs(lims_v), abs(pak_v), 1e-9)
            severe = diff > float(dq["source_disagreement_abs"]) and rel > float(
                dq["source_disagreement_rel"]
            )
            disagreement = {
                "lims": lims_v,
                "pak": pak_v,
                "abs": diff,
                "rel": rel,
                "severe": bool(severe),
                "resolution": "LIMS is the control fact (ЛИМС > ПАК > ВАК)",
            }
            if severe:
                flags.append(f"SOURCE_CONFLICT:|LIMS-PAK|={diff:.1f}")

        critical_ok = (
            (lims_health == "OK" or pak_health == "OK")
            and lims_health != "IMPLAUSIBLE"
            and not (disagreement and disagreement["severe"])
            and mode == "RUNNING"
            and tel_age <= float(dq["telemetry_max_age_minutes"])
            and missing_share <= dq["telemetry_missing_max_share"]
        )

        if self.lims_long is not None:
            available = self.lims_long[
                self.lims_long.timestamp <= t - pd.Timedelta(minutes=dq["lims_delay_minutes"])
            ]
            f = available["data_quality_flags"].fillna("")
            self._global_flags = {
                "nonnumeric": int(f.str.contains("NONNUMERIC").sum()),
                "duplicate_timestamps": int(f.str.contains("DUPLICATE").sum()),
            }
        assessment = DataQualityAssessment(
            decision_timestamp=t,
            sources=sources,
            flags=flags,
            telemetry_missing_share=missing_share,
            telemetry_stale_tags=stale_tags[:20],
            duplicate_timestamps=self._global_flags["duplicate_timestamps"],
            nonnumeric_values=self._global_flags["nonnumeric"],
            source_disagreement=disagreement,
            critical_data_ok=bool(critical_ok),
            notes=notes,
        )

        controls = {}
        for ctrl in cfg.controls["controls"]:
            vals = [
                float(row[tag])
                for tag in ctrl["tags"]
                if tag in row.index and np.isfinite(row[tag])
            ]
            if vals:
                controls[ctrl["canonical_name"]] = float(np.sum(vals)) if len(vals) > 1 else vals[0]

        return ProcessState(
            timestamp=t,
            telemetry={k: (float(v) if np.isfinite(v) else None) for k, v in row.items()},
            controls=controls,
            quality_sources={
                "lims": lims_status.to_dict(),
                "pak": pak_status.to_dict(),
                "disagreement": disagreement,
                "mode": mode,
                "telemetry_timestamp": str(tel_ts),
            },
            data_quality=assessment,
        )
