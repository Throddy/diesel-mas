"""Closed-loop comparison of configurations C and E on the independent plant.

The historical grid in `scripts/arch_compare.py` answers what each
configuration would say.  This answers what happens when its advice is
followed: the plant is the independent kinetic model, identical for both runs,
so the only difference is the response model the optimizer reasons with.

  C  the shipped system, response model M with physical sign constraints
  E  the same system with the response model fitted the ordinary way

Both are run on the risk scenario and on the breach scenario, with the
do-nothing trajectory kept as the reference line.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.config import load_config, project_root
from src.models.response_model import ResponseModel
from src.pipeline import build_system
from src.sim.closed_loop import ClosedLoopSimulator, SimConfig

SCENARIOS = {"риск по качеству": "quality_risk", "превышение предела": "limit_breach"}
UNCONSTRAINED = "response_model_unconstrained.joblib"


def simulator_for(cfg, sim_cfg, response: ResponseModel | None) -> ClosedLoopSimulator:
    """Build the system, optionally swapping the model the optimizer uses."""
    system = build_system(cfg)
    if response is not None:
        agent = system.orchestrator.oa.whatif
        assert hasattr(agent, "model"), "модель отклика переехала: поправьте подстановку"
        agent.model = response
    return ClosedLoopSimulator(system, sim_cfg)


def main(steps: int = 12, seed: int = 42) -> dict:
    cfg = load_config()
    models = project_root() / cfg.main["paths"]["models"]
    free_path = models / UNCONSTRAINED
    if not free_path.is_file():
        from scripts.train_response_unconstrained import main as train_free

        train_free()
    free = ResponseModel.load(free_path)
    diagnostics = json.loads((models / "response_model.json").read_text(encoding="utf-8"))
    sim_cfg = SimConfig(steps=steps, lag_minutes=float(diagnostics["lag_minutes"]), seed=seed)

    variants = {"C": None, "E": free}
    report = {
        "limit_mgkg": cfg.sulfur_limit,
        "steps": steps,
        "plant": "независимая кинетическая модель, src/sim/plant_model.py",
        "variants": {
            "C": "сданная конфигурация, модель отклика M со знаковыми ограничениями",
            "E": "та же система с моделью отклика без ограничений знаков и априори",
        },
        "temperature_log_sensitivity": {
            "M": diagnostics["temperature_log_sensitivity_joint"],
            "E": free.diagnostics["temperature_log_sensitivity_joint"],
        },
        "scenarios": {},
    }
    for label, scenario in SCENARIOS.items():
        start = pd.Timestamp(cfg.main["demo"][scenario])
        rows = {}
        reference = simulator_for(cfg, sim_cfg, None)
        rows["удержание режима"] = reference.run(start, "do_nothing").summary(cfg.sulfur_limit)
        for name, response in variants.items():
            result = simulator_for(cfg, sim_cfg, response).run(start, "mas")
            rows[name] = result.summary(cfg.sulfur_limit)
            rows[name]["headlines"] = sorted({r.headline for r in result.records if r.deciding})
            rows[name]["n_deciding_steps"] = sum(1 for r in result.records if r.deciding)
        report["scenarios"][label] = {"start": str(start), "policies": rows}

    out = project_root() / cfg.main["paths"]["reports"] / "closed_loop_architectures.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for label, block in report["scenarios"].items():
        print(f"\n{label}, старт {block['start']}")
        print(
            f"  {'политика':18s} {'вне спец., мин':>14s} {'действий':>9s} "
            f"{'сера в начале':>13s} {'сера в конце':>13s}"
        )
        for name, row in block["policies"].items():
            print(
                f"  {name:18s} {row['minutes_off_spec']:14d} {row['n_actions']:9d} "
                f"{row['sulfur_start_mgkg']:13.2f} {row['sulfur_final_mgkg']:13.2f}"
            )
    print(f"\n-> {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="замкнутый контур для конфигураций C и E")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    main(**vars(ap.parse_args()))
