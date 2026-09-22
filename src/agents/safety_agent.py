"""SafetyAgent: an independent constraint layer that issues verdicts.

The agent judges; it does not act.  It reads the candidate set and returns
per-candidate verdicts plus any cycle-wide veto.  Applying those verdicts to
the working set is the orchestrator's job, so no agent reaches into an object
another agent owns (K-23).

A veto is not overridable: the orchestrator may only narrow the feasible set
after a safety verdict, never widen it.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.config import load_config
from src.mas.messages import Verdict
from src.optimization.candidates import control_specs
from src.optimization.constraints import all_pass, check_candidate
from src.schemas import (
    CandidateAction,
    ConstraintCheck,
    DataQualityAssessment,
    ProcessState,
    QualityAssessment,
    ReliabilityAssessment,
)


class SafetyAgent:
    name = "SafetyAgent"

    def __init__(self, cfg=None, min_confidence: float = 0.25):
        self.cfg = cfg or load_config()
        self.min_confidence = self.cfg.dq("min_confidence")
        self.bounds: Dict[str, Tuple[float, float]] = {}
        for spec in control_specs(self.cfg):
            lo, hi = spec.bounds()
            if lo is not None:
                self.bounds[spec.canonical_name] = (lo, hi)

    def _cycle_vetoes(
        self, state: ProcessState, quality: QualityAssessment
    ) -> tuple[List[str], bool, bool]:
        """Conditions that disqualify the whole cycle, not one candidate."""
        dq: DataQualityAssessment = state.data_quality
        severe_conflict = bool(dq.source_disagreement and dq.source_disagreement["severe"])
        pak_failed = any(
            s.source == "PAK" and s.metric == "Mg.Sulfur" and s.health == "FAILED"
            for s in dq.sources
        )
        lims_stale = any(s.source == "LIMS" and s.health == "SUSPECT" for s in dq.sources)
        data_ok = (
            bool(dq.critical_data_ok) and not severe_conflict and not (pak_failed and lims_stale)
        )
        model_confident = quality.confidence >= self.min_confidence

        vetoes: List[str] = []
        if not data_ok:
            vetoes.append("critical input data are incomplete, stale or non-numeric")
        if severe_conflict:
            vetoes.append(
                "severe LIMS/PAK conflict: the laboratory value is the control fact "
                "and the online analyser is suspect"
            )
        if pak_failed and lims_stale:
            vetoes.append("online analyser failed (flatline) and the laboratory value is stale")
        if not model_confident:
            vetoes.append(
                f"forecast confidence {quality.confidence:.2f} below the minimum "
                f"{self.min_confidence:.2f}"
            )
        return vetoes, data_ok, model_confident

    def _severity_ok(self, cand: CandidateAction, base: float, tolerance: float) -> bool:
        """A move passes when it does not raise the severity proxy above tolerance.

        Holding the regime is always admissible: the system advises, it cannot
        stop the plant.  An unknown severity (failed sensor) blocks every move,
        and that case is also caught by HC-05 out-of-distribution.
        """
        if cand.is_do_nothing:
            return True
        if not np.isfinite(base):
            return False
        candidate_severity = cand.reliability_severity
        if candidate_severity is None or not np.isfinite(candidate_severity):
            return False
        return candidate_severity <= base + tolerance

    def run(
        self,
        cands: Sequence[CandidateAction],
        state: ProcessState,
        quality: QualityAssessment,
        reliability: ReliabilityAssessment,
    ) -> dict:
        """Return checks, verdicts and vetoes; leave every candidate untouched."""
        vetoes, data_ok, model_confident = self._cycle_vetoes(state, quality)
        base_severity = reliability.severity_index
        tolerance = float(self.cfg.main["optimization"]["severity_increase_tolerance"])

        checks_by_action: Dict[str, List[ConstraintCheck]] = {}
        verdicts: List[Verdict] = []
        for cand in cands:
            checks = check_candidate(
                cand,
                cfg=self.cfg,
                bounds=self.bounds,
                ood_ok=not reliability.ood_flag,
                data_ok=data_ok,
                reliability_ok=self._severity_ok(cand, base_severity, tolerance),
                model_confident=model_confident,
            )
            checks_by_action[cand.action_id] = checks
            failures = [
                f"{c.constraint_id} {c.name}: {c.detail}" for c in checks if c.status == "FAIL"
            ]
            admissible = all_pass(checks) and not vetoes
            verdicts.append(
                Verdict(
                    agent=self.name,
                    action_id=cand.action_id,
                    admissible=admissible,
                    dimension="safety",
                    reason=(
                        "; ".join(failures) if failures else ("; ".join(vetoes) if vetoes else "")
                    ),
                    detail={
                        "failures": failures,
                        "vetoes": list(vetoes),
                        "checks": [c.to_dict() for c in checks],
                    },
                )
            )

        return {
            "checks": {k: [c.to_dict() for c in v] for k, v in checks_by_action.items()},
            "veto": vetoes,
            "bounds": self.bounds,
            "severity_now": None if not np.isfinite(base_severity) else float(base_severity),
            "severity_class_now": reliability.severity_class,
            "verdicts": verdicts,
            "admissible_ids": [v.action_id for v in verdicts if v.admissible],
        }
