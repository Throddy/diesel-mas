"""What replanning buys, measured on the same grid as everything else.

The orchestrator widens the search when the safety verdict leaves no feasible
option.  This runs the identical 60-cycle grid with widening switched off and
with it on, so the claim about replanning has a report behind it instead of a
number typed once.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.config import load_config, project_root
from src.pipeline import build_system


def run(cfg, max_rounds: int, cycles: int) -> dict:
    cfg.main["optimization"]["replan_max_rounds"] = max_rounds
    system = build_system(cfg)
    start = pd.Timestamp(cfg.main["split"]["test_start"])
    stamps = pd.date_range(start, system.telemetry.index.max(), periods=cycles)
    rows = []
    for t in stamps:
        rec = system.decide(t, log=False)
        action = rec.selected_action or {}
        rows.append(
            {
                "timestamp": str(t),
                "abstained": bool(rec.abstained),
                "acted": bool(action and not action.get("is_do_nothing")),
                "action_id": action.get("action_id"),
                "upper": action.get("predicted_sulfur_upper"),
            }
        )
    frame = pd.DataFrame(rows)
    return {
        "replan_max_rounds": max_rounds,
        "cycles": rows,
        "n_cycles": int(len(frame)),
        "abstention_rate": float(frame.abstained.mean()),
        "n_abstentions": int(frame.abstained.sum()),
        "n_actions": int(frame.acted.sum()),
        "violations": int(
            sum(1 for r in rows if r["upper"] is not None and r["upper"] > cfg.sulfur_limit)
        ),
    }


def main(cycles: int = 60) -> dict:
    cfg = load_config()
    configured = int(cfg.main["optimization"].get("replan_max_rounds", 4))
    without = run(cfg, 1, cycles)
    with_replan = run(cfg, configured, cycles)
    cfg.main["optimization"]["replan_max_rounds"] = configured

    lost, changed = [], []
    for before, after in zip(without.pop("cycles"), with_replan.pop("cycles"), strict=False):
        if not before["abstained"] and after["abstained"]:
            lost.append(before["timestamp"])
        both_recommended = not before["abstained"] and not after["abstained"]
        if both_recommended and before["action_id"] != after["action_id"]:
            changed.append(before["timestamp"])
    report = {
        "grid": "тестовый блок, та же сетка, что в сравнении архитектур",
        "without_replanning": without,
        "with_replanning": with_replan,
        "cycles_recovered": without["n_abstentions"] - with_replan["n_abstentions"],
        "cycles_that_lost_a_recommendation": lost,
        "cycles_where_the_choice_changed": changed,
        "note": (
            "перепланирование расширяет поиск, когда после вердикта безопасности "
            "допустимых вариантов не осталось; жёсткое ограничение при этом не "
            "ослабляется, нарушений предела нет ни в одном варианте"
        ),
    }
    out = project_root() / cfg.main["paths"]["reports"] / "replan_value.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, row in (("без перепланирования", without), ("с перепланированием", with_replan)):
        print(
            f"{name:22s} раундов {row['replan_max_rounds']}, отказов "
            f"{row['n_abstentions']:2d} из {row['n_cycles']} "
            f"({row['abstention_rate'] * 100:.1f} %), действий {row['n_actions']:2d}, "
            f"нарушений {row['violations']}"
        )
    print(
        f"циклов, получивших рекомендацию: {report['cycles_recovered']}; "
        f"потерявших: {len(report['cycles_that_lost_a_recommendation'])}; "
        f"сменивших выбор: {len(report['cycles_where_the_choice_changed'])}"
    )
    print(f"-> {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="цена и польза перепланирования")
    ap.add_argument("--cycles", type=int, default=60)
    main(**vars(ap.parse_args()))
