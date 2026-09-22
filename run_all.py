"""One-command end-to-end run: prepare, train, evaluate, tests, demos.

Everything is reproducible from the raw files the organisers supplied.  The
periods quoted in the documents are listed under ``reporting.periods`` in the
configuration and are computed on every run, each with the cycle count the
documents name, so a clean clone reproduces the published tables.  A further
period may be given with ``--start/--end`` (R-MEET-17), so the jury can ask
for any window and get a report for it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
sys.path.insert(0, str(ROOT))


def reporting_periods() -> list[dict]:
    from src.config import load_config

    return list(load_config().main.get("reporting", {}).get("periods", []))


def steps(start: str | None, end: str | None, cycles: int) -> list[tuple[str, list[str]]]:
    """The ordered pipeline; the period back-test is added only when asked for."""
    plan = [
        ("подготовка данных", [PY, "-m", "scripts.prepare_data"]),
        ("обучение моделей", [PY, "-m", "scripts.train"]),
        ("модели по горизонтам", [PY, "-m", "scripts.train_horizons"]),
        ("софт-сенсоры ВАК", [PY, "-m", "scripts.train_soft_sensors"]),
        ("графики отчёта", [PY, "-m", "scripts.make_report_plots"]),
        ("оценка на holdout", [PY, "-m", "scripts.evaluate", "--cycles", str(cycles)]),
        (
            "поиск демонстрационных моментов",
            [PY, "-m", "scripts.find_demo_moments", "--step-hours", "12", "--write-config"],
        ),
        ("тесты", [PY, "-m", "scripts.run_tests"]),
        ("smoke test", [PY, "-m", "scripts.smoke_test"]),
        ("модель отклика без ограничений", [PY, "-m", "scripts.train_response_unconstrained"]),
        ("сравнение архитектур", [PY, "-m", "scripts.arch_compare"]),
        ("оценка горизонтов", [PY, "-m", "scripts.eval_horizons"]),
        ("цена перепланирования", [PY, "-m", "scripts.replan_value"]),
        ("калибровка вероятности превышения", [PY, "-m", "scripts.risk_calibration"]),
        ("сравнение правил решения", [PY, "-m", "scripts.compare_decision_rules"]),
        ("замкнутый контур: риск", [PY, "-m", "scripts.simulate", "--scenario", "quality-risk"]),
        (
            "замкнутый контур: превышение",
            [PY, "-m", "scripts.simulate", "--scenario", "limit-breach"],
        ),
        ("замкнутый контур: архитектуры", [PY, "-m", "scripts.simulate_architectures"]),
    ]
    for period in reporting_periods():
        plan.append(
            (
                f"расчёт за период {period['start']} по {period['end']}",
                [
                    PY,
                    "-m",
                    "scripts.backtest",
                    "--start",
                    period["start"],
                    "--end",
                    period["end"],
                    "--cycles",
                    str(period.get("cycles", cycles)),
                ],
            )
        )
    if start and end:
        plan.append(
            (
                f"расчёт за период {start} по {end}",
                [
                    PY,
                    "-m",
                    "scripts.backtest",
                    "--start",
                    start,
                    "--end",
                    end,
                    "--cycles",
                    str(cycles),
                ],
            )
        )
    plan += [
        ("demo: стабильный режим", [PY, "-m", "scripts.demo", "--scenario", "stable"]),
        ("demo: риск по качеству", [PY, "-m", "scripts.demo", "--scenario", "quality-risk"]),
        ("demo: недостоверные данные", [PY, "-m", "scripts.demo", "--scenario", "bad-data"]),
        ("манифест прогона", [PY, "-m", "src.manifest"]),
        ("рендер опубликованных цифр", [PY, "-m", "scripts.render_results"]),
    ]
    return plan


def main(
    data_dir: str | None = None,
    skip_train: bool = False,
    start: str | None = None,
    end: str | None = None,
    cycles: int = 24,
) -> int:
    env = dict(os.environ)
    if data_dir:
        resolved = Path(data_dir).expanduser().resolve()
        if not resolved.is_dir():
            print(f"каталог исходных данных не найден: {resolved}")
            return 2
        env["DIESEL_MAS_DATA_DIR"] = str(resolved)
        print(f"исходные данные: {resolved}")
    if bool(start) != bool(end):
        print("укажите --start и --end вместе")
        return 2
    started = time.time()
    for title, cmd in steps(start, end, cycles):
        if skip_train and "обучение" in title:
            print(f"--- пропуск: {title}")
            continue
        print(f"\n=== {title}: {' '.join(cmd[1:])}", flush=True)
        step_started = time.time()
        code = subprocess.call(cmd, cwd=ROOT, env=env)
        print(f"--- {title}: код возврата {code}, {time.time() - step_started:.1f} с")
        if code != 0:
            print(f"ОСТАНОВ на шаге «{title}»")
            return code
    print(f"\nВсё выполнено за {time.time() - started:.0f} с. Дашборд:  streamlit run app.py")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="полный прогон решения")
    ap.add_argument(
        "--data-dir", default=None, help="каталог с файлами организаторов (по умолчанию data/raw)"
    )
    ap.add_argument("--skip-train", action="store_true", help="не переобучать модели")
    ap.add_argument("--start", default=None, help="начало отчётного периода, например 2026-01-01")
    ap.add_argument("--end", default=None, help="конец отчётного периода")
    ap.add_argument("--cycles", type=int, default=24, help="циклов решения при оценке")
    args = ap.parse_args()
    raise SystemExit(main(args.data_dir, args.skip_train, args.start, args.end, args.cycles))
