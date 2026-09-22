"""Render every published figure from the report files.

No number in README or docs is typed by hand: this script reads the JSON
reports produced by the pipeline and writes ``reports/results_summary.md``
plus ``reports/results.json``, which the README and the improvement report
quote verbatim.  If a report is missing, the corresponding line says so
instead of showing a stale value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config, project_root
from src.manifest import build_manifest

MISSING = "нет отчёта"


def _read(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value, digits=2, suffix="") -> str:
    if value is None:
        return "нет"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return f"{value}{suffix}"
    return f"{value:.{digits}f}{suffix}".replace(".", ",")


def _pct(value, digits=0) -> str:
    return "нет" if value is None else f"{value * 100:.{digits}f} %".replace(".", ",")


def collect(cfg) -> dict:
    """Gather the published figures from the report files that exist."""
    reports = project_root() / cfg.main["paths"]["reports"]
    evaluation = _read(reports / "evaluation_report.json")
    timing = _read(reports / "evaluation_timing.json")
    demo = _read(reports / "demo_moments.json")
    response = _read(project_root() / cfg.main["paths"]["models"] / "response_model.json")
    audit = _read(reports / "data_audit.json")
    out = {
        "model": {},
        "system": {},
        "response": {},
        "demo": {},
        "data": {},
        "periods": {},
        "architectures": {},
        "closed_loop": {},
        "forecast_improvement": {},
    }
    if evaluation:
        test = evaluation["model"]["test"]
        out["model"] = {
            "period": test["period"],
            "n": test["regression"]["n"],
            "MAE": test["regression"]["MAE"],
            "RMSE": test["regression"]["RMSE"],
            "R2": test["regression"]["R2"],
            "baseline_previous_lims_MAE": test["baselines"]["previous_lims"]["MAE"],
            "baseline_pak_MAE": test["baselines"]["pak_nowcast"]["MAE"],
            "baseline_train_median_MAE": test["baselines"]["train_median"]["MAE"],
            "recall_upper_bound": test["violation_upper_bound"].get("recall"),
            "precision_upper_bound": test["violation_upper_bound"].get("precision"),
            "interval_coverage": test["interval"].get("coverage"),
        }
        out["system"] = dict(evaluation["system"]["summary"])
        out["system"].update(timing or {})
        out["system"]["deterministic"] = evaluation["reproducibility"][
            "identical_recommendation_for_identical_input"
        ]
    if response:
        out["response"] = {
            "temperature_data": response["temperature_log_sensitivity_data"],
            "temperature_prior": response["temperature_log_sensitivity_prior"],
            "temperature_joint": response["temperature_log_sensitivity_joint"],
            "bootstrap": response["temperature_sensitivity_bootstrap"],
            "n_train": response["n_train"],
        }
    if demo:
        out["demo"] = {
            "block": demo["block"],
            "n_found": demo["n_found"],
            "selected": {k: v["timestamp"] for k, v in demo["selected"].items()},
            "period": demo["period"],
        }
    if audit:
        out["data"] = audit if isinstance(audit, dict) else {}
    for path in sorted((reports / "periods").glob("*/backtest.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        out["periods"][path.parent.name] = {
            "period": data["period"],
            "blocks": data["blocks_touched"],
            "model_MAE": data["model"].get("regression", {}).get("MAE"),
            "abstention_rate": data["system"].get("abstention_rate"),
            "action_rate_when_at_risk": data["system"].get("action_rate_when_at_risk"),
            "violations": data["system"].get("hard_constraint_violations_in_recommendations"),
        }
    architectures = _read(reports / "arch_compare.json")
    if architectures:
        out["architectures"] = {
            "cycles": architectures["cycles"],
            "rows": architectures["configurations"],
        }
    free = _read(reports / "response_unconstrained.json")
    if free:
        out["architectures"]["unconstrained_response"] = {
            "coefficients": free["coefficients"],
            "sign_verdict": free["sign_verdict"],
            "n_signs_against_physics": free["n_signs_against_physics"],
            "temperature_log_sensitivity": free["temperature_log_sensitivity"],
        }
    horizons = _read(reports / "horizons_evaluation.json")
    if horizons:
        out["horizons"] = horizons
    sensors = _read(reports / "soft_sensors.json")
    if sensors:
        out["soft_sensors"] = sensors
    replan = _read(reports / "replan_value.json")
    if replan:
        out["replanning"] = {
            "without": replan["without_replanning"],
            "with": replan["with_replanning"],
            "cycles_recovered": replan["cycles_recovered"],
            "cycles_lost": len(replan["cycles_that_lost_a_recommendation"]),
            "cycles_changed": len(replan["cycles_where_the_choice_changed"]),
        }
    loop = _read(reports / "closed_loop_architectures.json")
    if loop:
        out["closed_loop"] = loop
    improvement = _read(reports / "forecast_improvement.json")
    if improvement:
        out["forecast_improvement"] = {
            "variants": improvement["variants"],
            "by_period": improvement["by_period"],
            "significance": improvement["significance"],
            "additivity": improvement["additivity"],
        }
    blocks = _read(reports / "feature_blocks.json")
    if blocks:
        out["forecast_improvement"]["blocks"] = blocks["blocks"]
    selection = _read(reports / "feature_selection.json")
    if selection:
        out["forecast_improvement"]["feature_selection"] = selection
    excluded = reports / "excluded_periods.csv"
    if excluded.is_file():
        out["excluded_periods_rows"] = sum(1 for _ in excluded.open(encoding="utf-8")) - 1
    return out


def render(results: dict, manifest: dict) -> str:
    """Format the gathered figures as the Markdown block the docs include."""
    model, system = results["model"], results["system"]
    response, demo = results["response"], results["demo"]
    lines = [
        "<!-- СГЕНЕРИРОВАНО scripts/render_results.py. Не редактировать вручную. -->",
        "# Результаты прогона",
        "",
        f"Коммит `{(manifest['code']['commit'] or 'нет git')[:8]}`, "
        f"сгенерировано {manifest['generated_utc']}, "
        f"рабочее дерево чистое {_fmt(manifest['code']['working_tree_clean'])}.",
        "",
        "Блок оценки участвовал в выборе конфигурации, поэтому приведённые "
        "значения не являются независимой оценкой обобщающей способности.",
        "",
        "## Прогноз содержания серы",
    ]
    if not model:
        lines.append(f"{MISSING}: reports/evaluation_report.json")
    else:
        lines += [
            f"Период с {model['period'][0]} по {model['period'][1]}, "
            f"{model['n']} лабораторных анализов.",
            "",
            "| Оценка | MAE, мг/кг |",
            "|---|---|",
            f"| Модель прогноза | **{_fmt(model['MAE'])}** |",
            f"| Предыдущий лабораторный анализ | {_fmt(model['baseline_previous_lims_MAE'])} |",
            f"| Поточный анализатор | {_fmt(model['baseline_pak_MAE'])} |",
            f"| Медиана обучающего блока | {_fmt(model['baseline_train_median_MAE'])} |",
            "",
            "| Показатель | Значение |",
            "|---|---|",
            f"| RMSE, мг/кг | {_fmt(model['RMSE'])} |",
            f"| R², безразмерный | {_fmt(model['R2'])} |",
            f"| Эмпирическое покрытие интервала, доля | {_fmt(model['interval_coverage'], 3)} |",
            "| Номинальный уровень интервала, доля | 0,90 |",
            f"| Полнота обнаружения превышений, доля | {_fmt(model['recall_upper_bound'], 3)} |",
            f"| Точность обнаружения превышений, доля | "
            f"{_fmt(model['precision_upper_bound'], 3)} |",
        ]
    lines += ["", "## Поведение системы"]
    if not system:
        lines.append(f"{MISSING}: reports/evaluation_report.json")
    else:
        lines += [
            "| Показатель | Значение |",
            "|---|---|",
            f"| Циклов решения | {system.get('n_cycles')} |",
            f"| Доля отказов | {_pct(system.get('abstention_rate'))} |",
            f"| Рекомендаций с верхней границей выше предела | "
            f"{_pct(system.get('hard_constraint_violation_rate_of_recommendations'))} |",
            f"| Доля отклонённых небезопасных вариантов | "
            f"{_pct(system.get('unsafe_candidates_rejected_share'))} |",
            f"| Среднее время цикла, с | {_fmt(system.get('inference_seconds_mean'))} |",
            f"| Худшее время цикла, с | {_fmt(system.get('inference_seconds_max'))} |",
            f"| Одинаковый вход даёт одинаковую рекомендацию | "
            f"{_fmt(system.get('deterministic'))} |",
            "",
            "Нарушением считается выданная рекомендация, у которой верхняя граница "
            "прогноза выбранного варианта превышает предел. Проверка выполняется "
            "моделью прогноза на исторических данных.",
        ]
    lines += ["", "## Чувствительность серы к температуре по модели отклика"]
    if not response:
        lines.append(f"{MISSING}: models/response_model.json")
    else:
        lines += [
            "| Источник оценки | d ln S / dT, 1/°C |",
            "|---|---|",
            f"| Только данные, замкнутый контур | {_fmt(response['temperature_data'], 4)} |",
            f"| Физическое априори | {_fmt(response['temperature_prior'], 4)} |",
            f"| Совместная оценка, используется | **{_fmt(response['temperature_joint'], 4)}** |",
            "",
            f"Блочный бутстреп, 90 %: от {_fmt(response['bootstrap'][0], 4)} до "
            f"{_fmt(response['bootstrap'][1], 4)}; обучающих точек: {response['n_train']}.",
            "Знак задан химией гидроочистки, величина получена из данных и априори. "
            "На замкнутом контуре "
            "эффект по данным занижен (R-MEET-12).",
        ]
    lines += ["", "## Демонстрационные моменты"]
    if not demo:
        lines.append(f"{MISSING}: reports/demo_moments.json")
    else:
        lines += [
            f"Моменты найдены автоматическим сканом блока оценки "
            f"(с {demo['period'][0]} по {demo['period'][1]}); "
            "подходящих моментов найдено "
            + ", ".join(f"{v} для сценария {k}" for k, v in demo["n_found"].items())
            + "."
        ]
        lines += [f"- {k}: {v}" for k, v in demo["selected"].items()]
    if results["periods"]:
        lines += [
            "",
            "## Расчёт за периоды",
            "",
            "| Период | Блоки | MAE, мг/кг | Отказы | Действие при риске | Нарушений |",
            "|---|---|---|---|---|---|",
        ]
        for row in results["periods"].values():
            lines.append(
                f"| с {row['period'][0][:10]} по {row['period'][1][:10]} | "
                f"{', '.join(row['blocks'])} | "
                f"{_fmt(row['model_MAE'])} | {_pct(row['abstention_rate'])} | "
                f"{_pct(row['action_rate_when_at_risk'])} | {row['violations']} |"
            )
    arch = results.get("architectures") or {}
    if arch.get("rows"):
        lines += [
            "",
            f"## Сравнение архитектурных конфигураций, {arch['cycles']} циклов",
            "",
            "| Конфигурация | Нарушений | Отказы | Действие при риске "
            "| Отклонено небезопасных | Действий | Против физического знака |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in arch["rows"]:
            against = row.get("action_against_physics_rate")
            lines.append(
                f"| {row['configuration']} | {row['hard_limit_violations']} | "
                f"{_pct(row['abstention_rate'])} | {_pct(row['action_rate_when_at_risk'])} | "
                f"{row['unsafe_candidates_rejected']} | {row['n_recommendations_with_action']} | "
                f"{'нет действий' if against is None else _pct(against)} |"
            )
        free = arch.get("unconstrained_response")
        if free:
            temperature = free["temperature_log_sensitivity"]
            lines += [
                "",
                "В конфигурации E модель отклика обучена без ограничений на "
                "знаки коэффициентов. Чувствительность d ln S / dT равна "
                f"{_fmt(temperature['unconstrained_E'], 4)} 1/°C против "
                f"{_fmt(temperature['joint_M'], 4)} у модели M. Физическому знаку не "
                f"соответствуют {free['n_signs_against_physics']} коэффициента из "
                f"{len(free['coefficients'])}.",
            ]
    horizons = results.get("horizons") or {}
    if horizons.get("horizons"):
        lines += [
            "",
            "## Прогноз на горизонты",
            "",
            f"{horizons['scheme']}. Базовая оценка, {horizons['baseline']}. "
            f"Уровень интервала {_fmt(horizons['interval_level'], 2)}.",
            "",
            "| Горизонт, ч | Сравнено строк | MAE, мг/кг | MAE базовой оценки, мг/кг "
            "| MedianAE, мг/кг | RMSE, мг/кг | R² | Покрытие, доля | Ширина, мг/кг "
            "| Brier | Brier базовой частоты | Полнота | Точность | Пригоден |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for row in horizons["horizons"]:
            if row.get("status") != "OK":
                lines.append(
                    f"| {_fmt(row['horizon_hours'], 0)} | "
                    + " | ".join(["нет"] * 12)
                    + f" | {row.get('status')} |"
                )
                continue
            skill = row.get("skill", {})
            lines.append(
                f"| {_fmt(row['horizon_hours'], 0)} | {row['n_compared']} | "
                f"{_fmt(row['MAE'], 3)} | {_fmt(row['baseline_previous_lims_MAE'], 3)} | "
                f"{_fmt(row['MedianAE'], 3)} | {_fmt(row['RMSE'], 3)} | "
                f"{_fmt(row['R2'], 3)} | {_fmt(row['interval_coverage'], 3)} | "
                f"{_fmt(row['interval_mean_width_mgkg'], 2)} | "
                f"{_fmt(row['brier_probability_above_limit'], 3)} | "
                f"{_fmt(row['brier_base_rate'], 3)} | "
                f"{_fmt(row['recall_upper_bound'], 3)} | "
                f"{_fmt(row['precision_upper_bound'], 3)} | "
                f"{_fmt(bool(skill.get('usable_for_decisions')))} |"
            )
        usable = horizons.get("usable_horizons") or []
        lines += [
            "",
            "Модель и базовая оценка считаются на одном наборе строк. Горизонт "
            "участвует в проверке действия, если его MAE ниже базовой оценки и "
            "Brier ниже Brier базовой частоты. "
            + (
                f"Критерий проходят горизонты {usable}."
                if usable
                else "Критерий не проходит ни один горизонт."
            ),
        ]
    sensors = results.get("soft_sensors") or {}
    if sensors.get("targets"):
        lines += [
            "",
            "## Софт-сенсоры показателей продукта гидроочистки",
            "",
            "| Показатель | Единица | Наблюдений | MAE | MAE базовой оценки "
            "| Покрытие, доля | Предел | Статус |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in sensors["targets"]:
            lines.append(
                f"| {row['metric']} | {row['unit']} | {row['n_observations']} | "
                f"{_fmt(row.get('MAE'), 3)} | "
                f"{_fmt(row.get('baseline_previous_value_MAE'), 3)} | "
                f"{_fmt(row.get('interval_coverage'), 3)} | "
                f"{_fmt(row.get('limit'), 1)} | {row['status']} |"
            )
        rule = sensors["rule"]
        lines += [
            "",
            rule[0].upper() + rule[1:] + ". Базовая оценка, предыдущий лабораторный "
            "анализ показателя, доступный на момент решения.",
        ]
    replan = results.get("replanning")
    if replan:
        lines += [
            "",
            "## Перепланирование",
            "",
            "| Конфигурация | Отказы | Действий | Нарушений |",
            "|---|---|---|---|",
            f"| Один раунд поиска | {_pct(replan['without']['abstention_rate'])} | "
            f"{replan['without']['n_actions']} | {replan['without']['violations']} |",
            f"| Четыре раунда поиска | {_pct(replan['with']['abstention_rate'])} | "
            f"{replan['with']['n_actions']} | {replan['with']['violations']} |",
            "",
            f"Рекомендацию получают {replan['cycles_recovered']} циклов, "
            f"теряют {replan['cycles_lost']}, меняют выбор "
            f"{replan['cycles_changed']}.",
        ]
    loop = results.get("closed_loop") or {}
    for label, block in (loop.get("scenarios") or {}).items():
        lines += [
            "",
            f"## Замкнутый контур, сценарий {label}",
            "",
            "| Политика | Вне спецификации, мин | Действий "
            "| Сера в начале, мг/кг | Сера в конце, мг/кг |",
            "|---|---|---|---|---|",
        ]
        for name, row in block["policies"].items():
            lines.append(
                f"| {name.capitalize()} | {row['minutes_off_spec']} | {row['n_actions']} | "
                f"{_fmt(row['sulfur_start_mgkg'])} | {_fmt(row['sulfur_final_mgkg'])} |"
            )
    improvement = results.get("forecast_improvement") or {}
    if improvement.get("variants"):
        lines += [
            "",
            "## Вклад изменений в точность прогноза",
            "",
            "| Вариант | MAE, мг/кг | MedianAE, мг/кг | R² " "| Покрытие, доля | Ширина, мг/кг |",
            "|---|---|---|---|---|---|",
        ]
        for name, row in improvement["variants"].items():
            lines.append(
                f"| {name.capitalize()} | {_fmt(row['MAE'], 3)} | {_fmt(row['MedianAE'], 3)} | "
                f"{_fmt(row['R2'])} | {_fmt(row['coverage'])} | {_fmt(row['mean_width'])} |"
            )
        add = improvement["additivity"]
        lines += [
            "",
            "Сумма отдельных вкладов "
            f"{_fmt(add['sum_of_separate_deltas_MAE'], 3)} мг/кг, совместный вклад "
            f"{_fmt(add['joint_delta_MAE'], 3)} мг/кг.",
        ]
    if "excluded_periods_rows" in results:
        lines += [
            "",
            "## Исключённые периоды",
            f"Строк в `reports/excluded_periods.csv`: {results['excluded_periods_rows']}. "
            "Каждая строка содержит причину исключения.",
        ]
    return "\n".join(lines) + "\n"


def main() -> Path:
    cfg = load_config()
    results = collect(cfg)
    manifest = build_manifest(cfg)
    reports = project_root() / cfg.main["paths"]["reports"]
    (reports / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    out = reports / "results_summary.md"
    out.write_text(render(results, manifest), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))
    return out


if __name__ == "__main__":
    argparse.ArgumentParser(description="рендер опубликованных цифр из отчётов").parse_args()
    main()
