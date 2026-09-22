"""ReliabilityAgent: operating-severity proxy (NOT a failure probability)."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.config import load_config
from src.models.reliability import ReliabilityModel
from src.schemas import ProcessState, ReliabilityAssessment


class ReliabilityAgent:
    name = "ReliabilityAgent"

    def __init__(self, model: ReliabilityModel, cfg=None):
        self.model = model
        self.cfg = cfg or load_config()

    def assess_state(self, telemetry: dict) -> dict:
        """Answer a severity question about a hypothetical operating point.

        The optimizer asks through this method instead of reaching into the
        model, so the journal records which state was evaluated.
        """
        row = pd.Series(
            {k: (np.nan if v is None else v) for k, v in telemetry.items()}, dtype=float
        )
        result = self.model.assess(row)
        return {
            "severity_index": float(result["severity_index"]),
            "severity_class": str(result["severity_class"]),
            "ood_score": float(result["ood_score"]),
            "ood_flag": bool(result["ood_flag"]),
        }

    def run(self, state: ProcessState) -> ReliabilityAssessment:
        row = pd.Series({k: (np.nan if v is None else v) for k, v in state.telemetry.items()})
        res = self.model.assess(row)
        if any(f.startswith("SENSOR_FAULT") for f in state.data_quality.flags):
            res["severity_class"] = "UNKNOWN"
            res["severity_index"] = float("nan")
            res["ood_flag"] = True
        extra: List[str] = []
        admissible = res["severity_class"] != "HIGH"
        if res["envelope_violations"]:
            extra.append(
                "do not move controls that are already outside the historical envelope: "
                + ", ".join(res["envelope_violations"])
            )
        if res["ood_flag"]:
            extra.append("state flagged out-of-distribution: candidate moves must stay minimal")
        parts = [f"{f['signal']} robust-z={f['robust_z']}" for f in res["top_factors"]]
        explanation = (
            f"severity proxy {res['severity_index']:.2f} -> class {res['severity_class']} "
            f"(warn>={self.model.severity_warn:.2f}, high>={self.model.severity_high:.2f}); "
            f"main contributors: {'; '.join(parts) if parts else 'none'}"
        )
        return ReliabilityAssessment(
            decision_timestamp=pd.Timestamp(state.timestamp),
            severity_index=(
                float(res["severity_index"]) if np.isfinite(res["severity_index"]) else float("nan")
            ),
            severity_class=str(res["severity_class"]),
            ood_score=float(res["ood_score"]) if np.isfinite(res["ood_score"]) else float("nan"),
            ood_flag=bool(res["ood_flag"]),
            top_factors=res["top_factors"],
            admissible=bool(admissible),
            extra_constraints=extra,
            explanation=explanation,
        )
