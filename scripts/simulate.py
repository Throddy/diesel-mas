"""Run the closed-loop simulation for one scenario and compare three policies.

Historical replay answers "what would the system say"; this answers "what
happens if the advice is followed".  The response is computed with the same
model M the optimizer uses, so the run compares decision policies, not
physics.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.config import load_config, project_root
from src.pipeline import build_system
from src.sim.closed_loop import ClosedLoopSimulator, SimConfig

POLICIES = ("do_nothing", "naive", "mas")


def main(
    scenario: str = "quality_risk",
    steps: int = 12,
    seed: int = 42,
    sensitivity_factor: float = 1.0,
    delay_factor: float = 1.0,
    out_name: str | None = None,
) -> dict:
    cfg = load_config()
    system = build_system(cfg)
    start = pd.Timestamp(cfg.main["demo"][scenario])
    response = json.loads(
        (project_root() / cfg.main["paths"]["models"] / "response_model.json").read_text(
            encoding="utf-8"
        )
    )
    sim_cfg = SimConfig(
        steps=steps,
        lag_minutes=float(response["lag_minutes"]),
        seed=seed,
        sensitivity_factor=sensitivity_factor,
        delay_factor=delay_factor,
    )
    simulator = ClosedLoopSimulator(system, sim_cfg)

    results = {policy: simulator.run(start, policy) for policy in POLICIES}
    report = {
        "scenario": scenario,
        "start": str(start),
        "limit_mgkg": cfg.sulfur_limit,
        "simulation": {
            "steps": sim_cfg.steps,
            "step_minutes": sim_cfg.step_minutes,
            "lag_minutes": sim_cfg.lag_minutes,
            "time_constant_minutes": sim_cfg.time_constant_minutes,
            "max_temp_step_c": sim_cfg.max_temp_step_c,
            "max_feed_step_tph": sim_cfg.max_feed_step_tph,
            "seed": sim_cfg.seed,
            "sensitivity_factor": sim_cfg.sensitivity_factor,
            "delay_factor": sim_cfg.delay_factor,
            "plant": simulator.plant.describe(),
            "caveat": (
                "установка моделируется независимой кинетической моделью "
                "(src/sim/plant_model.py), система по-прежнему видит только N и M; "
                "сама модель установки не валидирована, см. "
                "reports/plant_model_validation.json"
            ),
        },
        "summary": [results[p].summary(cfg.sulfur_limit) for p in POLICIES],
        "trajectories": {p: [vars(r) for r in results[p].records] for p in POLICIES},
    }
    out = (
        project_root() / cfg.main["paths"]["reports"] / (out_name or f"closed_loop_{scenario}.json")
    )
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in report["summary"]:
        print(
            f"{row['policy']:11s} вне спецификации {row['minutes_off_spec']:4d} мин · "
            f"действий {row['n_actions']:2d} · "
            f"сера {row['sulfur_start_mgkg']:.2f} → {row['sulfur_final_mgkg']:.2f} мг/кг"
        )
    print(f"\n-> {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="имитационная среда с замкнутым контуром")
    ap.add_argument(
        "--scenario",
        default="quality-risk",
        choices=["stable", "quality-risk", "bad-data", "limit-breach"],
    )
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--sensitivity-factor",
        type=float,
        default=1.0,
        help="во сколько раз установка чувствительнее к температуре, чем считает система",
    )
    ap.add_argument(
        "--delay-factor",
        type=float,
        default=1.0,
        help="во сколько раз длиннее транспортное запаздывание установки",
    )
    ap.add_argument("--out", default=None, help="имя файла отчёта")
    a = ap.parse_args()
    main(a.scenario.replace("-", "_"), a.steps, a.seed, a.sensitivity_factor, a.delay_factor, a.out)
