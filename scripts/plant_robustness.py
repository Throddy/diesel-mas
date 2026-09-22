"""Does the advice stay safe when the plant is not what the system believes?

The system reasons with models N and M.  The simulated plant is a different
model, and here it is distorted further: more and less temperature-sensitive
than assumed, and slower to respond.  The question is not whether the sulfur
target is reached but whether the recommendations remain safe when the
process disagrees with the model behind them.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.config import load_config, project_root
from src.pipeline import build_system
from src.sim.closed_loop import ClosedLoopSimulator, SimConfig

VARIANTS = (
    {"name": "как ожидает система", "sensitivity_factor": 1.0, "delay_factor": 1.0},
    {"name": "чувствительность выше в 1,5 раза", "sensitivity_factor": 1.5, "delay_factor": 1.0},
    {
        "name": "чувствительность ниже в 1,5 раза",
        "sensitivity_factor": 1 / 1.5,
        "delay_factor": 1.0,
    },
    {"name": "запаздывание вдвое длиннее", "sensitivity_factor": 1.0, "delay_factor": 2.0},
)


def main(scenario: str = "limit_breach", steps: int = 12, policy: str = "mas") -> dict:
    cfg = load_config()
    system = build_system(cfg)
    start = pd.Timestamp(cfg.main["demo"][scenario])
    response = json.loads(
        (project_root() / cfg.main["paths"]["models"] / "response_model.json").read_text(
            encoding="utf-8"
        )
    )
    limit = cfg.sulfur_limit

    rows = []
    for variant in VARIANTS:
        sim_cfg = SimConfig(
            steps=steps,
            lag_minutes=float(response["lag_minutes"]),
            sensitivity_factor=variant["sensitivity_factor"],
            delay_factor=variant["delay_factor"],
        )
        simulator = ClosedLoopSimulator(system, sim_cfg)
        result = simulator.run(start, policy)
        summary = result.summary(limit)
        trajectory = [r.sulfur_mgkg for r in result.records]
        rows.append(
            {
                "variant": variant["name"],
                "sensitivity_factor": variant["sensitivity_factor"],
                "delay_factor": variant["delay_factor"],
                "plant_d_ln_s_dt": simulator.plant.temperature_sensitivity(
                    9343.0, 368.15, 214.86, 3.675
                ),
                "n_actions": summary["n_actions"],
                "minutes_off_spec": summary["minutes_off_spec"],
                "sulfur_start": summary["sulfur_start_mgkg"],
                "sulfur_final": summary["sulfur_final_mgkg"],
                "sulfur_max": summary["sulfur_max_mgkg"],
                "sulfur_min": float(min(trajectory)),
                "overshoot_below_half_limit": float(min(trajectory) < limit / 2),
            }
        )

    report = {
        "scenario": scenario,
        "policy": policy,
        "start": str(start),
        "limit_mgkg": limit,
        "model_M_d_ln_s_dt": response["temperature_log_sensitivity_joint"],
        "variants": rows,
        "question": (
            "остаётся ли рекомендация безопасной, когда представление " "системы о процессе неверно"
        ),
    }
    out = project_root() / cfg.main["paths"]["reports"] / "plant_robustness.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows:
        print(
            f"{row['variant']:34s} действий {row['n_actions']:2d} | "
            f"вне спец. {row['minutes_off_spec']:4d} мин | "
            f"сера {row['sulfur_start']:.2f} -> {row['sulfur_final']:.2f} "
            f"(мин {row['sulfur_min']:.2f}, макс {row['sulfur_max']:.2f})"
        )
    print(f"\n-> {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="устойчивость к ошибке модели установки")
    ap.add_argument("--scenario", default="limit-breach")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--policy", default="mas", choices=["mas", "naive", "do_nothing"])
    a = ap.parse_args()
    main(a.scenario.replace("-", "_"), a.steps, a.policy)
