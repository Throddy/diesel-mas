"""Typed contracts exchanged between agents.

Plain dataclasses (no pydantic dependency) with explicit ``to_dict`` so that
every agent hand-off can be serialised into the decision log.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


def _clean(obj: Any) -> Any:
    """Make a value JSON-safe, keeping "no measurement" distinguishable.

    numpy scalars are unwrapped first and then checked for finiteness: doing
    it the other way round let ``np.float64('nan')`` through as a bare ``nan``
    token, which is not valid JSON and made the value unreadable downstream.
    """
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, (datetime,)):
        return obj.isoformat(sep=" ")
    if hasattr(obj, "isoformat"):
        return obj.isoformat(sep=" ")
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


class Serialisable:
    def to_dict(self) -> Dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class Observation(Serialisable):
    timestamp: Optional[datetime]
    process_unit: str
    sample_point: str
    canonical_metric: str
    raw_metric: str
    value: Optional[float]
    original_unit: str
    canonical_unit: str
    source: str
    data_quality_flags: List[str] = field(default_factory=list)
    analyzer_health: Optional[str] = None
    age_at_decision_min: Optional[float] = None
    original_reference: str = ""


@dataclass
class SourceStatus(Serialisable):
    source: str
    metric: str
    last_timestamp: Optional[datetime]
    value: Optional[float]
    unit: str
    age_minutes: Optional[float]
    staleness_ratio: Optional[float]
    health: str
    flags: List[str] = field(default_factory=list)


@dataclass
class DataQualityAssessment(Serialisable):
    decision_timestamp: datetime
    sources: List[SourceStatus]
    flags: List[str]
    telemetry_missing_share: float
    telemetry_stale_tags: List[str]
    duplicate_timestamps: int
    nonnumeric_values: int
    source_disagreement: Optional[Dict[str, Any]]
    critical_data_ok: bool
    notes: List[str] = field(default_factory=list)


@dataclass
class ProcessState(Serialisable):
    timestamp: datetime
    telemetry: Dict[str, float]
    controls: Dict[str, float]
    quality_sources: Dict[str, Any]
    data_quality: DataQualityAssessment
    features: Dict[str, float] = field(default_factory=dict)
    vak: Dict[str, Optional[float]] = field(default_factory=dict)


@dataclass
class QualityAssessment(Serialisable):
    decision_timestamp: datetime
    metric: str
    unit: str
    point_forecast: Optional[float]
    lower: Optional[float]
    upper: Optional[float]
    risk_exceed_limit: Optional[float]
    limit: float
    model_version: str
    baseline_forecast: Optional[float] = None
    vak_forecast: Optional[float] = None
    nowcast_source: str = "NONE"
    horizons: List[Dict[str, Any]] = field(default_factory=list)
    other_quality: Dict[str, Any] = field(default_factory=dict)
    top_risk_factors: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    flags: List[str] = field(default_factory=list)


@dataclass
class HorizonForecast(Serialisable):
    """Forecast for one horizon, with the moments that produced it."""

    horizon_hours: float
    prediction: Optional[float]
    interval_lower: Optional[float]
    interval_upper: Optional[float]
    interval_level: float
    probability_above_limit: Optional[float]
    training_cutoff: Optional[str]
    calibration_cutoff: Optional[str]
    feature_as_of_time: Optional[datetime]
    target_time: Optional[datetime]
    data_status: str
    unit: str = "мг/кг"
    model_version: str = ""


@dataclass
class ReliabilityAssessment(Serialisable):
    decision_timestamp: datetime
    severity_index: float
    severity_class: str
    ood_score: float
    ood_flag: bool
    top_factors: List[Dict[str, Any]]
    admissible: bool
    extra_constraints: List[str]
    explanation: str
    proxy_disclaimer: str = (
        "Operating-severity proxy derived from historical operating envelope; "
        "no equipment-failure labels exist in the provided package."
    )


@dataclass
class CandidateAction(Serialisable):
    action_id: str
    control: Optional[str]
    tags: List[str]
    current_value: Optional[float]
    proposed_value: Optional[float]
    delta: Optional[float]
    direction: str
    is_do_nothing: bool = False
    predicted_sulfur: Optional[float] = None
    predicted_sulfur_upper: Optional[float] = None
    predicted_sulfur_lower: Optional[float] = None
    risk_exceed_limit: Optional[float] = None
    production_proxy: Optional[float] = None
    energy_proxy: Optional[float] = None
    reliability_severity: Optional[float] = None
    ood_score: Optional[float] = None
    action_magnitude: float = 0.0
    feasible: bool = True
    rejection_reasons: List[str] = field(default_factory=list)
    moves: Dict[str, float] = field(default_factory=dict)
    moves_from: Dict[str, float] = field(default_factory=dict)
    ramp_minutes: float = 0.0
    response_supported: bool = True
    score: Optional[float] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class ConstraintCheck(Serialisable):
    constraint_id: str
    name: str
    status: str
    detail: str
    confirmed_requirement: bool


@dataclass
class OperatorRecommendation(Serialisable):
    decision_timestamp: datetime
    model_version: str
    abstained: bool
    headline: str
    problem: str
    selected_action: Optional[Dict[str, Any]]
    expected_effect: Dict[str, Any]
    constraint_checks: List[Dict[str, Any]]
    confidence: Dict[str, Any]
    explanation: List[str]
    alternatives: List[Dict[str, Any]]
    reasons: List[str]
    data_freshness: Dict[str, Any]
    agent_trace: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BlendConstraint(Serialisable):
    """Blending interface.

    The provided package contains no blending component data, so quantitative
    blend optimisation is disabled (see docs/assumptions_and_limitations.md,
    A-09).  The constraint itself is implemented and unit-tested.
    """

    component_names: List[str]
    tolerance: float = 1e-9

    def check(self, fractions: Dict[str, float]) -> ConstraintCheck:
        missing = [c for c in self.component_names if c not in fractions]
        if missing:
            return ConstraintCheck(
                "HC-02", "blend_fraction_sum", "FAIL", f"missing fractions for {missing}", True
            )
        if any(v < 0 for v in fractions.values()):
            return ConstraintCheck("HC-02", "blend_fraction_sum", "FAIL", "negative fraction", True)
        total = float(sum(fractions.values()))
        ok = abs(total - 1.0) <= self.tolerance
        return ConstraintCheck(
            "HC-02",
            "blend_fraction_sum",
            "PASS" if ok else "FAIL",
            f"sum(fractions)={total:.12f}",
            True,
        )
