"""Closed-loop simulation: the advice is applied and comes back as state.

Historical replay shows what the system says; it cannot show what happens
after the operator follows the advice.  This module closes that loop.  A
policy proposes a move, the move is applied to the controls under a rate
limit, the response model turns the new controls into a target sulfur, and the
simulated sulfur relaxes towards that target with a transport delay and a time
constant.  The next decision is taken on the updated state.

Nothing in the agents or the models is modified: the simulator owns a private
copy of the telemetry slice and writes the controls it has applied into it, so
the agents see the consequences of their own advice through the ordinary
DataView.

The plant is an independent kinetic model (``src/sim/plant_model.py``), not
model M.  The system still reasons with N and M, so the gap between what it
expects and what the simulated plant does is real model error, which is the
thing a closed-loop run is for.  The plant model is itself unvalidated - see
``reports/plant_model_validation.json`` - so a run shows robustness to being
wrong, not what this unit would actually do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class SimConfig:
    """Simulation parameters; every one of them is a stated assumption."""

    steps: int = 12
    step_minutes: int = 10
    settle_minutes: float = 900.0
    lag_minutes: float = 360.0
    time_constant_minutes: float = 180.0
    max_temp_step_c: float = 1.0
    max_feed_step_tph: float = 5.0
    seed: int = 42
    sensitivity_factor: float = 1.0
    delay_factor: float = 1.0


def build_default_plant(sim_cfg: "SimConfig"):
    """The kinetic plant, calibrated once and optionally distorted.

    ``sensitivity_factor`` scales the activation energy, which is what makes
    the plant respond more or less strongly to temperature than the system
    believes; ``delay_factor`` stretches the transport delay.
    """
    from dataclasses import replace as _replace

    from src.sim.plant_model import PlantModel, PlantParams

    params = PlantParams(
        delay_minutes=sim_cfg.lag_minutes * sim_cfg.delay_factor,
        time_constant_minutes=sim_cfg.time_constant_minutes,
    )
    plant = PlantModel(params).calibrate()
    if sim_cfg.sensitivity_factor != 1.0:
        distorted = _replace(
            plant.p,
            activation_energy_kj_mol=plant.p.activation_energy_kj_mol * sim_cfg.sensitivity_factor,
        )
        plant = PlantModel(distorted).calibrate(
            feed_sulfur_mgkg=plant.p.reference_feed_sulfur_mgkg,
            temperature_c=plant.p.reference_temperature_c,
            feed_tph=plant.p.reference_feed_tph,
            pressure_mpa=plant.p.reference_pressure_mpa,
            product_sulfur_mgkg=plant.p.reference_product_sulfur_mgkg,
        )
    return plant


@dataclass
class StepRecord:
    step: int
    timestamp: str
    sulfur_mgkg: float
    target_sulfur_mgkg: float
    controls: Dict[str, float]
    action: Optional[str]
    moved: bool
    severity: Optional[float]
    deciding: bool = True
    headline: Optional[str] = None


@dataclass
class PolicyResult:
    policy: str
    description: str
    records: List[StepRecord] = field(default_factory=list)

    def summary(self, limit: float) -> dict:
        sulfur = [r.sulfur_mgkg for r in self.records]
        off_spec = sum(1 for v in sulfur if v > limit)
        step_minutes = 10
        return {
            "policy": self.policy,
            "description": self.description,
            "steps": len(self.records),
            "minutes_off_spec": off_spec * step_minutes,
            "share_off_spec": off_spec / max(len(sulfur), 1),
            "n_actions": sum(1 for r in self.records if r.moved),
            "sulfur_start_mgkg": sulfur[0] if sulfur else None,
            "sulfur_final_mgkg": sulfur[-1] if sulfur else None,
            "sulfur_max_mgkg": max(sulfur) if sulfur else None,
            "margin_final_mgkg": None if not sulfur else round(limit - sulfur[-1], 3),
            "severity_final": self.records[-1].severity if self.records else None,
        }


class ClosedLoopSimulator:
    """Run one policy from a starting moment and record the trajectory."""

    TEMP = "ht:T5"
    FEED = "ht:F9"

    def __init__(self, system, sim_cfg: SimConfig, plant=None):
        self.system = system
        self.cfg = sim_cfg
        self.limit = float(system.cfg.sulfur_limit)
        self.reliability = system.orchestrator.ra.model
        self.plant = plant if plant is not None else build_default_plant(sim_cfg)

    def _initial_state(self, start: pd.Timestamp) -> tuple[dict, float]:
        """Controls and sulfur the run starts from, taken from the system."""
        recommendation = self.system.decide(start, log=False)
        telemetry = self.system.telemetry.loc[:start].iloc[-1]
        controls = {self.TEMP: float(telemetry[self.TEMP]), self.FEED: float(telemetry[self.FEED])}
        sulfur = float(recommendation.agent_trace["QualityAgent"]["point_forecast"])
        return controls, sulfur

    def _target_sulfur(
        self, base_controls: dict, controls: dict, base_sulfur: float, feed_sulfur: float
    ) -> float:
        """Steady state the plant would reach, through the kinetic model.

        The plant is anchored to the starting point so a run begins where the
        real regime was: the kinetic model supplies the response to a change,
        not the absolute level, which it is not accurate enough to set.
        """
        at_base = self.plant.steady_state(
            feed_sulfur, base_controls[self.TEMP], base_controls[self.FEED], self._pressure
        )
        at_now = self.plant.steady_state(
            feed_sulfur, controls[self.TEMP], controls[self.FEED], self._pressure
        )
        if at_base <= 0:
            return base_sulfur
        return float(base_sulfur * at_now / at_base)

    def _relax(self, current: float, target: float, elapsed_minutes: float) -> float:
        """Dynamics of the plant: transport delay, then a first-order lag."""
        return self.plant.step(current, target, elapsed_minutes, self.cfg.step_minutes)

    def _apply(self, controls: dict, desired: dict) -> tuple[dict, bool]:
        """Move controls towards the desired values under the rate limits."""
        limits = {self.TEMP: self.cfg.max_temp_step_c, self.FEED: self.cfg.max_feed_step_tph}
        moved = False
        out = dict(controls)
        for tag, target in desired.items():
            if tag not in out or target is None or not np.isfinite(target):
                continue
            step = float(np.clip(target - out[tag], -limits[tag], limits[tag]))
            if abs(step) > 1e-9:
                out[tag] = out[tag] + step
                moved = True
        return out, moved

    def _severity(self, controls: dict, row: pd.Series) -> Optional[float]:
        state = row.copy()
        for tag, value in controls.items():
            state[tag] = value
        try:
            return float(self.reliability.assess(state)["severity_index"])
        except Exception:
            return None

    def run(self, start: pd.Timestamp, policy: str) -> PolicyResult:
        """Simulate ``steps`` decisions under one policy."""
        np.random.seed(self.cfg.seed)
        descriptions = {
            "do_nothing": "удержание режима: уставки не меняются",
            "naive": "наивное правило: +1 °C за шаг, пока прогноз выше предела",
            "mas": "мультиагентная система: рекомендация с проверками и вето",
        }
        result = PolicyResult(policy, descriptions[policy])
        telemetry = self.system.telemetry
        row = telemetry.loc[:start].iloc[-1]
        self._pressure = float(row["ht:P3"])
        feed_sulfur = self._feed_sulfur(start)
        base_controls, base_sulfur = self._initial_state(start)
        controls, sulfur = dict(base_controls), base_sulfur
        applied: list[tuple[float, str]] = []

        settle_steps = int(self.cfg.settle_minutes // self.cfg.step_minutes)
        for step in range(self.cfg.steps + settle_steps):
            t = start + pd.Timedelta(minutes=self.cfg.step_minutes * step)
            deciding = step < self.cfg.steps
            if deciding:
                minute = self.cfg.step_minutes * step
                recent = self._recent_moves(applied, minute)
                desired, action_text, headline = self._decide(policy, t, controls, sulfur, recent)
            else:
                desired, action_text, headline = {}, None, "наблюдение эффекта"
            controls, moved = self._apply(controls, desired)
            if moved and deciding:
                for control in self._control_names(desired):
                    applied.append((self.cfg.step_minutes * step, control))
            target = self._target_sulfur(base_controls, controls, base_sulfur, feed_sulfur)
            sulfur = self._relax(sulfur, target, self.cfg.step_minutes * step)
            result.records.append(
                StepRecord(
                    step=step,
                    timestamp=str(t),
                    sulfur_mgkg=round(sulfur, 4),
                    target_sulfur_mgkg=round(target, 4),
                    controls={k: round(v, 3) for k, v in controls.items()},
                    action=action_text,
                    moved=moved,
                    deciding=deciding,
                    severity=self._severity(controls, row),
                    headline=headline,
                )
            )
        return result

    def _recent_moves(self, applied, minute_now: float):
        """Moves already made, as the decision layer expects to see them."""
        from src.decision.expected_loss import RecentMove

        return [
            RecentMove(control=control, minutes_ago=minute_now - minute, delta=0.0)
            for minute, control in applied
        ]

    def _control_names(self, desired: dict) -> list[str]:
        """Canonical control names behind the telemetry tags a move touches."""
        mapping = {self.TEMP: "ht_r201_gss_outlet_temp", self.FEED: "ht_feed_flow_mass"}
        return [mapping[tag] for tag in desired if tag in mapping]

    def _decide(
        self, policy: str, t: pd.Timestamp, controls: dict, sulfur: float, recent=None
    ) -> tuple[dict, Optional[str], Optional[str]]:
        """What the policy wants to do at this step."""
        if policy == "do_nothing":
            return {}, None, "Ничего не делать"
        if policy == "naive":
            if sulfur > self.limit:
                return (
                    {self.TEMP: controls[self.TEMP] + self.cfg.max_temp_step_c},
                    f"+{self.cfg.max_temp_step_c} °C",
                    "Поднять температуру",
                )
            return {}, None, "Порог не превышен"
        recommendation = self.system.decide(t, log=False, recent_moves=recent)
        action = recommendation.selected_action
        if not action or action.get("is_do_nothing"):
            return {}, None, recommendation.headline
        desired = dict(self._moves(action).items())
        return desired, recommendation.headline, recommendation.headline

    def _moves(self, action: dict) -> dict:
        """Translate a recommendation into target values per telemetry tag."""
        mapping = {"ht_r201_gss_outlet_temp": self.TEMP, "ht_feed_flow_mass": self.FEED}
        out = {}
        for control, value in (action.get("moves") or {}).items():
            tag = mapping.get(control)
            if tag:
                out[tag] = float(value)
        return out

    def _feed_sulfur(self, t: pd.Timestamp) -> float:
        series = self.system.orchestrator.oa.qa.lims_other.get("feed_sulfur")
        if series is None or not len(series):
            return float("nan")
        available = series.loc[:t]
        return float(available.iloc[-1]) if len(available) else float("nan")
