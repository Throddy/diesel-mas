"""Single source of operator-facing Russian recommendation text."""

from __future__ import annotations

from src.mas.policies import REPLAN_ROUNDS
from src.schemas import OperatorRecommendation

HEALTH = {
    "OK": "исправен",
    "FAILED": "отказ",
    "SUSPECT": "сомнителен",
    "MISSING": "нет данных",
    "IMPLAUSIBLE": "неправдоподобен",
}
MODE = {
    "RUNNING": "рабочий режим",
    "TRANSIENT": "переходный режим",
    "SHUTDOWN": "останов",
    "SENSOR_FAULT": "отказ датчика давления",
}
CHECKS = {
    "PRODUCT-T95": "T95 продукта гидроочистки по формуле ВАК",
    "PRODUCT-CN": "Цетановое число",
    "HC-01": "Сера по верхней границе",
    "HC-02": "Сумма долей блендинга",
    "HC-03": "Модельный диапазон, темп и применимость",
    "HC-04": "Достоверность данных",
    "HC-05": "Область обучения",
    "HC-06": "Тяжесть режима",
}


def number(value, digits=2):
    return "нет данных" if value is None else f"{value:.{digits}f}".replace(".", ",")


def reasons_ru(state, quality, reliability, candidates):
    """Named reasons for withholding a recommendation, in the operator language."""
    reasons = []
    mode = state.quality_sources.get("mode", "RUNNING")
    if mode != "RUNNING":
        reasons.append(MODE.get(mode, mode) + ": управляющие действия заблокированы.")
    if not state.data_quality.critical_data_ok:
        reasons.append(
            "Критические данные недостоверны; проверьте статусы источников " "и датчиков."
        )
    disagreement = state.data_quality.source_disagreement
    if disagreement and disagreement["severe"]:
        reasons.append("Сильное расхождение ЛИМС и ПАК; требуется сверка измерений.")
    if reliability.ood_flag:
        reasons.append("Режим за пределами области обучения модели.")
    if reliability.severity_class == "UNKNOWN":
        reasons.append("Тяжесть режима не определена: отказ датчика.")
    if not any(c.feasible for c in candidates):
        movable = [c for c in candidates if not c.is_do_nothing]
        if all(not c.response_supported for c in movable):
            reasons.append(
                "Нет применимого отклика: проверьте наличие свежей серы "
                "сырья и диапазоны управляющих."
            )
        if all((c.predicted_sulfur_upper or float("inf")) > quality.limit for c in candidates):
            reasons.append(
                "Ни один вариант не обеспечивает ограничение по сере " "по верхней границе."
            )
    return reasons or ["Нет варианта, проходящего все проверки SafetyAgent."]


STATUS_TEXT = {
    "INSUFFICIENT_HISTORY": "доступных остатков калибровки меньше требуемого минимума",
    "NO_AVAILABILITY_TIMES": "у остатков калибровки нет времени доступности",
    "NO_TARGET": "нет цели для этого горизонта",
    "NO_SKILL": "модель не обходит базовую оценку и в ограничениях не участвует",
}


def _status_lines(quality) -> list:
    """Report every state that leaves a value undefined, without substituting one."""
    other = quality.other_quality or {}
    lines = []
    interval = other.get("interval_status")
    if interval and interval != "OK":
        lines.append(
            f"Интервал прогноза не построен: {STATUS_TEXT.get(interval, interval)}. "
            "Ограничение по верхней границе не подтверждается."
        )
    unusable = [
        f"{row['horizon_hours']:.0f} ч, {STATUS_TEXT.get(row['data_status'], row['data_status'])}"
        for row in (quality.horizons or [])
        if row.get("data_status") != "OK"
    ]
    if unusable:
        lines.append("Горизонты без прогноза: " + "; ".join(unusable) + ".")
    horizon = other.get("decision_horizon") or {}
    if horizon.get("source") == "nowcast":
        lines.append(
            "Проверка действия идёт по верхней границе прогноза текущего режима. "
            f"{horizon.get('reason', '')}".strip()
        )
    weak = [row["metric"] for row in (other.get("soft_sensors") or []) if row.get("status") != "OK"]
    if weak:
        lines.append(
            "Показатели без подтверждённого навыка модели и потому без оценки: "
            + ", ".join(weak)
            + "."
        )
    return lines


def _message_id(rec, agent: str) -> str:
    """Identifier of the message a block of numbers came from."""
    return (rec.agent_trace.get("message_ids", {}) or {}).get(agent, "не назначено")


class ExplainerAgent:
    name = "ExplainerAgent"

    def __init__(self, cfg):
        self.cfg = cfg
        self.controls = {c["canonical_name"]: c for c in cfg.controls["controls"]}

    def action_text(self, action, state=None):
        if action is None:
            return "Надёжной рекомендации нет"
        if action.is_do_nothing:
            return "Ничего не делать, сохранить текущий режим"
        parts = []
        for key, value in action.moves.items():
            control = self.controls[key]
            current = state.controls.get(key) if state else None
            parts.append(
                f"{control['description_ru']}: {number(current)} "
                f"на {number(value)} {control['unit']}"
            )
        return "; ".join(parts)

    def build(self, t, state, quality, reliability, candidates, ranked, safety, trace):
        selected = ranked[0] if ranked and not safety["veto"] else None
        reasons = [] if selected else reasons_ru(state, quality, reliability, candidates)
        best = min(
            (c for c in candidates if not c.is_do_nothing and c.response_supported),
            key=lambda c: c.predicted_sulfur_upper,
            default=None,
        )
        direction = None
        if selected is None and state.quality_sources.get("mode") == "RUNNING" and best:
            direction = {
                "action": self.action_text(best, state),
                "upper": best.predicted_sulfur_upper,
                "rejections": best.rejection_reasons,
                "notice": "Направление для технолога, не разрешённое действие",
            }
        freshness = {
            source.source + ":" + source.metric: source.to_dict()
            for source in state.data_quality.sources
        }
        freshness["TELEMETRY"] = {
            "last_timestamp": state.quality_sources["telemetry_timestamp"],
            "mode": state.quality_sources["mode"],
        }
        effect = {
            "product_release_authorized": False,
            "scope": "Совет по гидроочистке; соответствие товарного ДТ не установлено",
            "best_direction": direction,
        }
        if selected:
            effect.update(
                sulfur_before=quality.point_forecast,
                sulfur_forecast=selected.predicted_sulfur,
                sulfur_interval=[selected.predicted_sulfur_lower, selected.predicted_sulfur_upper],
                production_proxy=selected.production_proxy,
                energy_proxy=selected.energy_proxy,
                reliability_severity=selected.reliability_severity,
                ramp_minutes=selected.ramp_minutes,
                effect_delay_minutes=self.cfg.main["response"]["lag_minutes"],
            )
        checks = safety["checks"].get(selected.action_id if selected else "A00_do_nothing", [])
        checks = checks + self._product_spec_checks(quality)
        explain = [
            "Совет требует согласования с технологом: исполнительные контуры ГО "
            "не подтверждены отдельной схемой.",
            "Качество и ограничения проверены до экономического ранжирования. "
            "Ранжирование использует фиксированные шкалы.",
            "Интервал прогноза и бутстреп эффекта не являются гарантией на замкнутом контуре.",
        ]
        if selected and selected.is_do_nothing:
            explain.append(
                "Текущее состояние проходит модельные проверки; "
                "вмешательство не улучшает стоимость действия."
            )
        elif selected:
            explain.append(
                "Выбрано минимальное вмешательство среди вариантов, " "прошедших проверки."
            )
        recovery = (
            "Повторить расчёт после устранения указанных причин; не применять отклонённые варианты."
        )
        explain += _status_lines(quality)
        explain.append(recovery)
        return OperatorRecommendation(
            t,
            quality.model_version,
            selected is None,
            self.action_text(selected, state),
            f"Сера ГО: {number(quality.point_forecast)} мг/кг; верхняя граница "
            f"{number(quality.upper)} мг/кг.",
            selected.to_dict() if selected else None,
            effect,
            checks,
            {
                "overall": quality.confidence,
                "terms": quality.other_quality.get("confidence_terms", {}),
                "ood_flag": reliability.ood_flag,
                "severity_class": reliability.severity_class,
            },
            explain,
            [c.to_dict() for c in ranked[1:4]],
            reasons,
            freshness,
            trace,
        )

    def _product_spec_checks(self, quality):
        """Product specification from the concept document: sulfur, T95, cetane.

        Sulfur is the hard constraint and is checked by SafetyAgent (HC-01).
        T95 is checked through the organiser formula 24-2000:GODT:T95, a model
        estimate rather than a laboratory result.  Cetane number is not checked:
        nothing in the package measures or estimates it.
        """
        limit = float(self.cfg.main["quality"].get("t95_limit_c", 360.0))
        vak = (quality.other_quality or {}).get("vak", {})
        t95 = vak.get("24-2000:GODT:T95")
        if t95 is None:
            t95_check = {
                "constraint_id": "PRODUCT-T95",
                "name": "T95 ≤ 360 °C",
                "status": "NOT_VERIFIED",
                "detail": "формула ВАК для T95 не рассчитана на этот момент",
                "confirmed_requirement": True,
            }
        else:
            t95_check = {
                "constraint_id": "PRODUCT-T95",
                "name": "T95 ≤ 360 °C",
                "status": "PASS" if t95 <= limit else "FAIL",
                "detail": f"оценка по формуле ВАК {number(t95)} °C против "
                f"{number(limit)} °C, источник предела "
                "Ustanovka_AVT_merged.pdf лист 4; оценка модельная, "
                "лабораторным анализом не является",
                "confirmed_requirement": True,
            }
        return [
            {
                "constraint_id": "HC-02",
                "name": "Сумма долей блендинга",
                "status": "NOT_VERIFIED",
                "detail": "нет данных товарного резервуара и состава смеси",
                "confirmed_requirement": True,
            },
            t95_check,
            {
                "constraint_id": "PRODUCT-CN",
                "name": "Цетановое число ≥ 51",
                "status": "NOT_VERIFIED",
                "detail": "цетановое число не измеряется и не оценивается ни "
                "одним источником пакета",
                "confirmed_requirement": True,
            },
        ]

    def card(self, rec):
        """Render the operator card: Russian text, units, one number per source."""
        q = rec.agent_trace["QualityAgent"]
        effect = rec.expected_effect
        ids = rec.agent_trace.get("message_ids", {})

        def ref(agent):
            return f" [сообщение {ids.get(agent, 'не назначено')}]"

        mode = MODE.get(rec.data_freshness["TELEMETRY"]["mode"], "неизвестный режим")
        lines = [
            f"**{rec.decision_timestamp} · {mode}**",
            "",
            "**Измерения серы**" + ref("DataQualityAgent"),
        ]
        lines += self._measurement_lines(rec.data_freshness)
        lines += [
            "",
            "**Риск**" + ref("QualityAgent"),
            rec.problem,
            f"Интервал {number(q['lower'])}–{number(q['upper'])} мг/кг "
            "(номинальный уровень 90 %, эмпирическое покрытие на тесте ниже "
            "номинала — интервал занижает неопределённость, см. "
            "reports/evaluation_report.json); "
            f"эмпирическая оценка риска > 10: {number((q['risk_exceed_limit'] or 0) * 100, 0)}%.",
            "",
            "**Действие**" + ref("OptimizationAgent"),
            rec.headline,
            "",
            "**Ожидаемый эффект**" + ref("OptimizationAgent"),
        ]
        lines += self._effect_lines(rec, effect)
        lines += self._risk_economy_lines(rec)
        lines += self._replan_lines(rec)
        if rec.confidence.get("severity_class") == "HIGH":
            lines += [
                "",
                "**Внимание: тяжесть режима в критическом классе** (прокси, допущение). "
                "Разрешены только варианты, не ухудшающие тяжесть; удержание режима допустимо.",
            ]
        lines += ["", "**Ограничения**" + ref("SafetyAgent")]
        lines += self._constraint_lines(rec.constraint_checks)
        lines += [
            "",
            "**Уверенность**" + ref("QualityAgent"),
            f"{number(rec.confidence['overall'] * 100, 0)}% — эвристический индекс доверия, "
            "не вероятность правильности.",
            self._effect_evidence_line(),
            "",
            "**Почему так и что изменит решение**",
            *rec.explanation,
            "",
            effect["scope"] + ". Выпуск товарного продукта не разрешается этой карточкой.",
        ]
        return "\n\n".join(lines)

    LABELS = {
        ("LIMS", "Mg.Sulfur"): "ЛИМС, сера продукта",
        ("PAK", "Mg.Sulfur"): "ПАК, сера продукта",
        ("LIMS", "Feed.Sulfur"): "ЛИМС, сера сырья (возмущение)",
    }

    def _measurement_lines(self, freshness):
        """Sulfur measurements, product first, feed - the main disturbance - after."""
        out = []
        for key, source in freshness.items():
            if key == "TELEMETRY":
                continue
            label = self.LABELS.get((source.get("source"), source.get("metric")))
            if label is None:
                continue
            if source.get("value") is None:
                out.append(f"- {label}: нет анализа до момента решения.")
                continue
            hours = (source["age_minutes"] or 0) / 60.0
            value = f"{number(source['value'])} {source['unit']}"
            if source.get("metric") == "Feed.Sulfur":
                value += f" ({number(source['value'] / 10000.0, 3)} % масс.)"
            out.append(
                f"- {label}: {value}, возраст {number(hours, 1)} ч; "
                f"{HEALTH.get(source['health'], source['health'])}."
            )
        return out or ["- измерений серы до момента решения нет."]

    def _effect_lines(self, rec, effect):
        """What the selected action changes, or why there is no action."""
        if not rec.selected_action:
            out = list(rec.reasons)
            direction = effect.get("best_direction")
            if direction:
                out.append(
                    "Лучшее направление для технолога (не разрешённое действие): "
                    + direction["action"]
                    + f"; расчётная верхняя граница "
                    f"{number(direction['upper'])} мг/кг — предела не достигает."
                )
            return out
        return [
            f"Сера {number(effect['sulfur_before'])} → {number(effect['sulfur_forecast'])} мг/кг; "
            f"верхняя граница {number(effect['sulfur_interval'][1])} мг/кг. "
            f"Модельное запаздывание {effect['effect_delay_minutes']} мин; "
            f"выполнение изменения за {number(effect['ramp_minutes'], 0)} мин.",
            f"Выпуск: {number(effect['production_proxy'])} т/ч; "
            f"тепловой прокси: {number(effect['energy_proxy'])}; "
            f"тяжесть режима: {number(effect['reliability_severity'])}.",
        ]

    def _risk_economy_lines(self, rec):
        """State the decision in terms of risk and what each option costs.

        The operator is asked to act on a probability, so the card names it:
        how likely the limit is to be breached if nothing changes, what that
        expectation costs against what the proposed move costs, and why one
        was preferred.  Prices are dimensionless proxies - the package has no
        economics - and the card says so.
        """
        from src.decision.expected_loss import LossWeights, intervention_cost, off_spec_loss

        trace = rec.agent_trace.get("OptimizationAgent") or {}
        candidates = trace.get("candidates") or []
        hold = next((c for c in candidates if c.get("is_do_nothing")), None)
        if hold is None or hold.get("risk_exceed_limit") is None:
            return []
        weights = LossWeights.from_config(self.cfg)

        class _Row:
            def __init__(self, data):
                self.__dict__.update(data)

        hold_row = _Row(hold)
        hold_loss = off_spec_loss(hold_row, weights)
        lines = [
            "",
            "**Риск и цена решения**" + f" [сообщение {_message_id(rec, 'OptimizationAgent')}]",
            f"Если режим не менять, вероятность выйти за 10 мг/кг равна "
            f"{number((hold.get('risk_exceed_limit') or 0) * 100, 0)} %; "
            f"ожидаемая цена бездействия {number(hold_loss)} в безразмерных единицах.",
        ]

        action = rec.selected_action
        if action and not action.get("is_do_nothing"):
            row = _Row(action)
            move_cost = intervention_cost(row, hold_row, weights)
            move_risk = off_spec_loss(row, weights)
            lines.append(
                f"Предлагаемое действие снижает вероятность до "
                f"{number((action.get('risk_exceed_limit') or 0) * 100, 0)} %: цена риска "
                f"{number(move_risk)} плюс цена вмешательства {number(move_cost)}, "
                f"итого {number((move_risk or 0) + move_cost)}. Действуем, потому что это "
                f"дешевле бездействия."
            )
        elif action:
            lines.append(
                "Удерживаем режим: запас до предела достаточен, а вмешательство стоит "
                "энергии, выпуска и ресурса оборудования и риска почти не снимает."
            )
        lines.append(
            "Цены безразмерные: экономических данных в пакете нет, это прокси "
            "для сравнения вариантов между собой."
        )
        return lines

    def _replan_lines(self, rec):
        """Say whether the search was widened, and what that produced.

        A recommendation found only after widening rests on a larger move than
        the ordinary search allows, and an abstention after three rounds means
        the space really was searched.  Both belong in front of the operator.
        """
        rounds = (rec.agent_trace.get("Orchestrator") or {}).get("rounds") or []
        if len(rounds) < 2:
            return []
        last = rounds[-1]
        extra = len(rounds) - 1
        word = "раунд" if extra == 1 else "раунда" if extra < 5 else "раундов"
        head = (
            f"\n**Перепланирование**: выполнен{'' if extra == 1 else 'о'} {extra} "
            f"дополнительный {word} из {len(REPLAN_ROUNDS) - 1}."
            if extra == 1
            else f"\n**Перепланирование**: выполнено {extra} дополнительных {word} "
            f"из {len(REPLAN_ROUNDS) - 1}."
        )
        steps = "; ".join(
            f"раунд {r['round']} ({r['widening']}): вариантов {r['n_candidates']}, "
            f"допустимых {r['n_feasible']}"
            for r in rounds
        )
        if last["selected"]:
            tail = (
                "Решение найдено только после расширения поиска, поэтому предложенное "
                "изменение крупнее обычного шага."
            )
        else:
            tail = (
                "Допустимых вариантов не нашлось ни в одном раунде, поэтому система "
                "отказывается от рекомендации."
            )
        return [head, steps + ". " + tail]

    def _effect_evidence_line(self) -> str:
        """On what the claimed effect of a move rests.

        The magnitude comes from a physical prior mixed with closed-loop data.
        Episodes where an operator stepped a control and held it would be
        direct evidence; the search found none on this unit, and the card says
        so rather than implying the effect was verified here.
        """
        import json

        from src.config import project_root

        path = project_root() / self.cfg.main["paths"]["reports"] / "natural_experiments.json"
        if not path.is_file():
            return (
                "Эффект воздействия: априори и данные замкнутого контура; "
                "натуральные эпизоды не проверялись."
            )
        try:
            episodes = json.loads(path.read_text(encoding="utf-8"))["conclusion"]["episodes"]
        except (KeyError, ValueError):
            return (
                "Эффект воздействия: априори и данные замкнутого контура; "
                "отчёт по эпизодам нечитаем."
            )
        if episodes:
            return (
                f"Эффект воздействия подтверждён {episodes} историческими эпизодами "
                "с удержанием режима."
            )
        return (
            "Эффект воздействия опирается на физическое априори и данные замкнутого "
            "контура: подтверждающих исторических эпизодов с удержанием режима не "
            "найдено (0), установка работает под регулятором."
        )

    def _constraint_lines(self, checks):
        """One line per constraint: several sub-checks share an id, the worst wins."""
        order, worst = [], {}
        rank = {"FAIL": 0, "NOT_VERIFIED": 1, "NA": 2, "PASS": 3}
        for check in checks:
            cid = check["constraint_id"]
            if cid not in worst:
                order.append(cid)
                worst[cid] = check
            elif rank[check["status"]] < rank[worst[cid]["status"]]:
                worst[cid] = check
        status_ru = {
            "PASS": "выполнено",
            "FAIL": "НЕ ВЫПОЛНЕНО",
            "NA": "не применимо",
            "NOT_VERIFIED": "не проверено",
        }
        out = []
        for cid in order:
            check = worst[cid]
            name = CHECKS.get(cid, check["name"])
            note = "" if check["confirmed_requirement"] else " (допущение, не промышленный предел)"
            detail = f" — {check['detail']}" if check["status"] in ("FAIL", "NOT_VERIFIED") else ""
            out.append(f"- {name}: {status_ru[check['status']]}{note}{detail}")
        return out
