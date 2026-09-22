"""Streamlit dashboard: АВТ -> гидроочистка -> блендинг advisory prototype.

Run with:  streamlit run app.py
The dashboard is advisory only - it never sends anything to the plant.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from src.agents.explainer_agent import ExplainerAgent
from src.config import load_config, project_root
from src.optimization.objectives import energy_formula_text
from src.pipeline import build_system

st.set_page_config(page_title="Diesel MAS - советчик оператора", layout="wide")


@st.cache_resource(show_spinner="Загрузка данных и моделей...")
def get_system():
    return build_system(load_config())


def sequence_diagram(journal) -> str:
    """Collapse the journal into a readable sequence of agent exchanges.

    A cycle asks two what-if questions per candidate and the two recipients
    alternate, so a naive run-length fold leaves hundreds of lines.  Any
    stretch of what-if traffic is therefore summarised as one block counting
    the questions each agent answered.
    """
    if not journal:
        return "_журнал пуст_"

    what_if = {"WhatIfRequest", "WhatIfResponse"}
    lines: list[str] = []
    pending: dict[str, int] = {}

    def flush() -> None:
        if pending:
            parts = ", ".join(f"{agent}: {count}" for agent, count in sorted(pending.items()))
            lines.append(f"OptimizationAgent <-> what-if ({parts} ответов)")
            pending.clear()

    for message in journal:
        if message["type"] in what_if:
            if message["type"] == "WhatIfResponse":
                pending[message["sender"]] = pending.get(message["sender"], 0) + 1
            continue
        flush()
        lines.append(f"{message['sender']} -> {message['recipient']}: {message['type']}")
    flush()
    return "```\n" + "\n".join(lines) + "\n```"


def _natural_experiments():
    """The natural-experiment report, or None when it has not been built."""
    path = project_root() / "reports" / "natural_experiments.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _measured_coverage():
    """Coverage actually achieved on the test block, or None if not evaluated yet."""
    path = project_root() / "reports" / "evaluation_report.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    return report.get("model", {}).get("test", {}).get("interval", {}).get("coverage")


system = get_system()
cfg = system.cfg
tel = system.telemetry
limit = cfg.sulfur_limit

st.title("Мультиагентная система управления производством ДТ")
st.caption("АВТ → гидроочистка → блендинг · режим исторического replay · рекомендательный прототип")

with st.sidebar:
    st.header("Момент решения")
    scenario = st.selectbox("Сценарий", ["stable", "quality_risk", "bad_data", "свой момент"])
    if scenario != "свой момент":
        default = pd.Timestamp(cfg.main["demo"][scenario])
        st.caption(cfg.main["demo"]["descriptions"][scenario])
    else:
        default = pd.Timestamp(cfg.main["demo"]["stable"])
    day = st.date_input(
        "Дата", default.date(), min_value=tel.index.min().date(), max_value=tel.index.max().date()
    )
    clock = st.time_input("Время", default.time())
    t = pd.Timestamp.combine(pd.Timestamp(day), clock)
    step = st.button("Шаг вперёд на 10 минут")
    if step:
        t = t + pd.Timedelta(minutes=10)
    st.divider()
    st.metric("Жёсткий предел по сере", f"{limit:.0f} мг/кг")
    st.caption("Единственное подтверждённое ТЗ ограничение по продукту")

rec = system.decide(t)
trace = rec.agent_trace

tabs = st.tabs(
    [
        "Обзор",
        "Качество",
        "Горизонты",
        "Качество данных",
        "Надёжность",
        "Рекомендация",
        "Альтернативы и Парето",
        "Трассировка агентов",
        "Обмен сообщениями",
    ]
)

with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.markdown("### АВТ\nК-1 / К-2 / К-10, печи П-1, П-3")
    c1.metric("Отбор фр.290-350", f"{tel.loc[:t, 'avt:F30'].iloc[-1]:.1f}")
    c2.markdown("### Гидроочистка 24-2000\nР-201 / Р-202, К-201")
    c2.metric("Температура ГСС Р-201", f"{tel.loc[:t, 'ht:T5'].iloc[-1]:.1f}")
    c3.markdown("### Блендинг\n**данных нет в пакете**")
    c3.caption(
        "Количественная оптимизация блендинга отключена (допущение A-09); "
        "интерфейс BlendConstraint реализован и покрыт тестом sum(fractions)=1"
    )
    st.divider()
    st.subheader("Текущие значения подтверждённых управляющих переменных")
    rows = []
    for ctrl in cfg.controls["controls"]:
        vals = [tel.loc[:t, tag].iloc[-1] for tag in ctrl["tags"] if tag in tel.columns]
        rows.append(
            {
                "управляющая": ctrl["canonical_name"],
                "теги": ", ".join(ctrl["tags"]),
                "описание": ctrl["description_ru"],
                "значение": round(float(np.nansum(vals)), 4) if vals else None,
                "единица": ctrl["unit"],
                "пром. предел известен": ctrl["industrial_limit_known"],
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

with tabs[1]:
    q = trace["QualityAgent"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Прогноз серы", f"{q['point_forecast']:.2f} мг/кг")
    c2.metric(
        "Верхняя граница интервала",
        f"{q['upper']:.2f}",
        delta=f"предел {limit:.0f}",
        delta_color="inverse",
    )
    c3.metric("Риск > предела", f"{q['risk_exceed_limit']:.0%}")
    c4.metric("Источник nowcast", q["nowcast_source"])
    coverage = _measured_coverage()
    if coverage is not None:
        st.warning(
            f"Интервал построен на номинальный уровень 90 %, но эмпирическое "
            f"покрытие на тестовом блоке — {coverage:.2f}. Интервал занижает "
            f"неопределённость; верхняя граница менее консервативна, чем заявлено."
        )
    hist = system.target.loc[:t].tail(120)
    pak_hist = system.pak.loc[t - pd.Timedelta(days=30) : t]
    chart = pd.DataFrame({"ЛИМС": hist})
    chart["предел"] = limit
    st.line_chart(chart)
    st.caption("ЛИМС (последние 120 анализов) и предел 10 мг/кг")
    st.line_chart(pd.DataFrame({"ПАК": pak_hist}))
    st.caption("ПАК за 30 суток до момента решения")

with tabs[2]:
    q = trace["QualityAgent"]
    other = q.get("other_quality") or {}
    lims_age = (q.get("other_quality") or {}).get("confidence_terms")
    age = trace["DataQualityAgent"]["sources"]
    lims_row = next((r for r in age if r["source"] == "LIMS"), None)
    if lims_row is not None and lims_row.get("age_minutes") is not None:
        st.metric(
            "Возраст последнего лабораторного анализа", f"{lims_row['age_minutes'] / 60:.1f} ч"
        )
    horizons = q.get("horizons") or []
    if not horizons:
        st.info("Модели по горизонтам не обучены: выполните python -m scripts.train_horizons")
    else:
        table = pd.DataFrame(
            [
                {
                    "горизонт, ч": row["horizon_hours"],
                    "целевой момент": row["target_time"],
                    "прогноз, мг/кг": row["prediction"],
                    "нижняя граница": row["interval_lower"],
                    "верхняя граница": row["interval_upper"],
                    "уровень интервала": row["interval_level"],
                    "вероятность выше предела": row["probability_above_limit"],
                    "статус": row["data_status"],
                }
                for row in horizons
            ]
        )
        st.dataframe(table, width="stretch", hide_index=True)
    chosen = other.get("decision_horizon") or {}
    if chosen:
        st.subheader("Горизонт проверки действия")
        st.write(f"Источник: {chosen.get('source')}. {chosen.get('reason')}")
    if rec.abstained and rec.reasons:
        st.subheader("Причина отсутствия рекомендации")
        for reason in rec.reasons:
            st.write(reason)
    sensors = other.get("soft_sensors") or []
    if sensors:
        st.subheader("Показатели продукта гидроочистки")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "показатель": row["metric"],
                        "единица": row["unit"],
                        "оценка": row["value"],
                        "нижняя": row["lower"],
                        "верхняя": row["upper"],
                        "предел": row["limit"],
                        "вид оценки": row["kind"],
                        "статус": row["status"],
                    }
                    for row in sensors
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Оценка обученной модели показателя. Лабораторным анализом не "
            "является. Показатель без подтверждённого предела в жёстких "
            "ограничениях не участвует."
        )

with tabs[3]:
    dq = trace["DataQualityAgent"]
    st.subheader("Источники")
    st.dataframe(pd.DataFrame(dq["sources"]), width="stretch", hide_index=True)
    st.subheader("Флаги")
    st.write(dq["flags"] or "флагов нет")
    if dq["source_disagreement"]:
        st.subheader("Расхождение источников")
        st.json(dq["source_disagreement"])
    st.subheader("Натуральные эксперименты")
    natural = _natural_experiments()
    if natural is None:
        st.caption("Отчёт не построен: `python -m scripts.natural_experiments`")
    else:
        found = natural["conclusion"]["episodes"]
        if found:
            rows = natural["controls"]["ht:T5"]["episodes"]
            frame = pd.DataFrame(
                [
                    {
                        "предсказано моделью M": r["predicted_log_change"],
                        "фактически": r["observed_log_change"],
                    }
                    for r in rows
                    if r.get("predicted_log_change") is not None
                ]
            )
            st.scatter_chart(frame, x="предсказано моделью M", y="фактически")
        else:
            st.warning(
                f"Подходящих эпизодов не найдено: {natural['conclusion']['statement']}. "
                "График «предсказано против факта» строить не на чем."
            )
            st.caption(natural["conclusion"]["why_none"])
            st.dataframe(
                pd.DataFrame(natural["controls"]["ht:T5"]["criterion_map"]).rename(
                    columns={
                        "hold_hours": "удержание, ч",
                        "tolerance": "допуск",
                        "episodes": "эпизодов",
                        "shorter_than_transport_delay": "короче запаздывания",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

    c1, c2, c3 = st.columns(3)
    c1.metric("Пропуски телеметрии", f"{dq['telemetry_missing_share']:.1%}")
    c2.metric("Дубли timestamp в ЛИМС", dq["duplicate_timestamps"])
    c3.metric("Нечисловые значения ЛИМС", dq["nonnumeric_values"])

with tabs[4]:
    r = trace["ReliabilityAgent"]
    c1, c2 = st.columns(2)
    c1.metric("Индекс тяжести режима (прокси)", f"{r['severity_index']:.2f}", r["severity_class"])
    c2.metric("OOD", "да" if r["ood_flag"] else "нет", f"score {r['ood_score']:.3f}")
    st.warning(r["proxy_disclaimer"])
    st.dataframe(pd.DataFrame(r["top_factors"]), width="stretch", hide_index=True)

with tabs[5]:
    if rec.abstained:
        st.error(f"### {rec.headline}")
    else:
        st.success(f"### {rec.headline}")
    st.markdown(ExplainerAgent(cfg).card(rec))
    with st.expander("Приложение для инженера: проверки, единицы, источник цифр"):
        checks = pd.DataFrame(rec.constraint_checks).rename(
            columns={
                "constraint_id": "ограничение",
                "name": "проверка",
                "status": "статус",
                "detail": "пояснение",
                "confirmed_requirement": "подтверждено ТЗ",
            }
        )
        st.dataframe(checks, width="stretch", hide_index=True)
        st.caption(energy_formula_text())

with tabs[6]:
    cands = pd.DataFrame(trace["OptimizationAgent"]["candidates"])
    cols = [
        "action_id",
        "control",
        "direction",
        "current_value",
        "proposed_value",
        "predicted_sulfur",
        "predicted_sulfur_upper",
        "risk_exceed_limit",
        "production_proxy",
        "energy_proxy",
        "reliability_severity",
        "feasible",
        "score",
    ]
    st.dataframe(cands[[c for c in cols if c in cands.columns]], width="stretch", hide_index=True)
    st.caption("Парето-фронт: " + ", ".join(trace["OptimizationAgent"]["pareto_front"]))
    feas = cands[cands["feasible"]]
    if len(feas) > 1:
        st.scatter_chart(
            feas, x="energy_proxy", y="production_proxy", size="predicted_sulfur_upper"
        )
    st.subheader("Причины отклонения")
    for _, row in cands[~cands["feasible"]].iterrows():
        st.write(f"**{row['action_id']}** — {'; '.join(row['rejection_reasons'][:2])}")

with tabs[7]:
    st.markdown("`Data → Quality → Reliability → Optimizer → Safety → Orchestrator`")
    for agent in [
        "DataQualityAgent",
        "QualityAgent",
        "ReliabilityAgent",
        "OptimizationAgent",
        "SafetyAgent",
        "Orchestrator",
    ]:
        with st.expander(agent):
            st.json(trace[agent])
    st.caption(f"Каждый цикл дописывается в {cfg.main['decision_log']['path']}")

with tabs[8]:
    journal = trace.get("messages", [])
    rounds = trace["Orchestrator"].get("rounds", [])
    c1, c2, c3 = st.columns(3)
    c1.metric("Сообщений в цикле", len(journal))
    c2.metric("Раундов планирования", len(rounds))
    c3.metric("Оспоренных вариантов", len(trace["Orchestrator"].get("conflicts", [])))

    st.subheader("Диаграмма последовательности")
    st.caption(
        "Показаны обмены между агентами; повторяющиеся запросы what-if "
        "по каждому кандидату свёрнуты в один блок."
    )
    st.markdown(sequence_diagram(journal), unsafe_allow_html=False)

    if rounds:
        st.subheader("Раунды планирования")
        st.dataframe(
            pd.DataFrame(rounds).rename(
                columns={
                    "round": "раунд",
                    "widening": "расширение поиска",
                    "n_candidates": "вариантов",
                    "n_feasible": "допустимых",
                    "selected": "выбрано",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    conflicts = trace["Orchestrator"].get("conflicts", [])
    if conflicts:
        st.subheader("Разрешение конфликтов")
        st.caption(
            "Вариант прошёл проверки одного агента и был запрещён другим. "
            "Запрет может наложить любой агент, снять его не может никто; "
            "приоритет измерения определяет только, какой запрет назван ведущим."
        )
        st.dataframe(
            pd.DataFrame(conflicts)[
                ["action_id", "blocked_by", "dimension_ru", "passed_checks_of", "reasons"]
            ].rename(
                columns={
                    "action_id": "вариант",
                    "blocked_by": "запрет наложил",
                    "dimension_ru": "измерение",
                    "passed_checks_of": "прошёл проверки",
                    "reasons": "причины",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    st.subheader("Журнал")
    table = pd.DataFrame(
        [
            {
                "id": m["id"],
                "от": m["sender"],
                "кому": m["recipient"],
                "тип": m["type"],
                "отвечает на": ", ".join(m["parent_ids"]),
            }
            for m in journal
        ]
    )
    st.dataframe(table, width="stretch", hide_index=True)
    chosen = st.selectbox("Показать содержимое сообщения", [m["id"] for m in journal])
    body = next(m for m in journal if m["id"] == chosen)
    st.json(json.loads(body["payload_json"]))
