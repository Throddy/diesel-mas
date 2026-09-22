"""Historical replay of the three mandatory demo scenarios.

Only information available at the replayed timestamp is used; the system does
not send anything to the plant - it is advisory.  Operator-facing text comes
from ExplainerAgent, the single source of Russian wording (K-25); this script
only adds the technical appendix an engineer needs to audit a card.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.agents.explainer_agent import ExplainerAgent, number
from src.config import load_config
from src.pipeline import build_system

STATUS_RU = {
    "PASS": "выполнено",
    "FAIL": "НЕ ВЫПОЛНЕНО",
    "NA": "не применимо",
    "NOT_VERIFIED": "не проверено",
}


def block_of(t: pd.Timestamp, cfg) -> str:
    """Which chronological block the replayed moment belongs to."""
    split = cfg.main["split"]
    if t <= pd.Timestamp(split["train_end"]):
        return "ОБУЧЕНИЕ"
    if t <= pd.Timestamp(split["valid_end"]):
        return "ВАЛИДАЦИЯ"
    return "ТЕСТ"


def print_recommendation(rec, cfg, explainer: ExplainerAgent) -> None:
    """Print the operator card, then the engineer's audit appendix."""
    t = pd.Timestamp(rec.decision_timestamp)
    print("=" * 78)
    print(
        f"РЕШЕНИЕ {rec.decision_timestamp} · блок {block_of(t, cfg)} · модель {rec.model_version}"
    )
    print("=" * 78)
    print(explainer.card(rec))
    print("-" * 78)
    print("ПРИЛОЖЕНИЕ ДЛЯ ИНЖЕНЕРА")
    print("  проверки безопасности:")
    for check in rec.constraint_checks:
        mark = "" if check["confirmed_requirement"] else "  (допущение, не промышленный предел)"
        print(
            f"    [{STATUS_RU[check['status']]}] {check['constraint_id']} {check['name']}: "
            f"{check['detail']}{mark}"
        )
    if rec.alternatives:
        print("  ближайшие альтернативы:")
        for alt in rec.alternatives:
            print(
                f"    {alt['action_id']:32s} оценка={alt['score']:.3f} "
                f"верхняя граница={number(alt['predicted_sulfur_upper'])} мг/кг "
                f"выпуск={number(alt['production_proxy'])} т/ч"
            )
    print(f"  доверие: {rec.confidence['overall']:.2f} · составляющие: {rec.confidence['terms']}")
    print(f"  сообщения шины: {rec.agent_trace['message_ids']}")
    print("=" * 78)


def main(scenario: str, timestamp: str | None = None, steps: int = 1) -> None:
    cfg = load_config()
    system = build_system(cfg)
    explainer = ExplainerAgent(cfg)
    demo = cfg.main["demo"]
    t = pd.Timestamp(timestamp) if timestamp else pd.Timestamp(demo[scenario])
    print(f"\n### СЦЕНАРИЙ: {demo['descriptions'][scenario]}")
    print(f"### момент выбран автоматически: {demo.get('selection', 'задан в конфигурации')}\n")
    for i in range(steps):
        print_recommendation(system.decide(t + pd.Timedelta(minutes=10 * i)), cfg, explainer)
    print(f"журнал решений дополнен: {cfg.main['decision_log']['path']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="stable", choices=["stable", "quality-risk", "bad-data"])
    ap.add_argument(
        "--timestamp",
        "--at",
        dest="timestamp",
        default=None,
        help="произвольный момент решения вместо сценарного",
    )
    ap.add_argument("--steps", type=int, default=1, help="прокрутить N шагов по 10 мин")
    a = ap.parse_args()
    main(a.scenario.replace("-", "_"), a.timestamp, a.steps)
