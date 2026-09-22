"""Final assembly: does the gain survive when the adopted pieces are combined?

Three ideas were tried separately.  Feature reduction was rejected, it won on
validation and lost on test.  Expanding retraining was adopted behind a flag.
The two feature blocks, controller activity and the catalyst index, were
adopted together.  Measured one at a time, each looked additive; this script
checks the pair on the same test block, because two changes that each lower
the error can still overlap.

The grid is fixed before the run and the test block is opened once for all
four cells.  Nothing here selects anything: the choices were already made in
`reports/feature_selection.json`, `reports/walk_forward.json` and
`reports/feature_blocks.json`.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from scripts.walk_forward import by_period, expanding_schedule, fixed_schedule, score
from src.config import load_config, project_root
from src.data.loaders import clean_telemetry
from src.data.store import load_df
from src.features.dataset import build_dataset_for_model, chronological_split
from src.models.quality import SulfurModel, calibrate_model, make_model

PERIODS = {"январь 2026": ("2026-01-01", "2026-02-01"), "май 2026": ("2026-05-01", "2026-06-01")}

GRID = {
    "базовый набор признаков": {"blocks": False, "expanding": False},
    "расширяющееся переобучение": {"blocks": False, "expanding": True},
    "блоки признаков": {"blocks": True, "expanding": False},
    "переобучение и блоки признаков": {"blocks": True, "expanding": True},
}

REPORTED = (
    "MAE",
    "MedianAE",
    "RMSE",
    "R2",
    "coverage",
    "mean_width",
    "recall",
    "precision",
    "n_fits",
)


def paired_bootstrap(errors_a, errors_b, seed: int, draws: int = 2000) -> dict:
    """Confidence interval for MAE(b) - MAE(a) on the same analyses.

    Resampling the analyses in pairs keeps the comparison on the same rows, so
    the interval answers whether the gap between two variants is larger than
    the noise of a 367-analysis test block.
    """
    rng = np.random.default_rng(seed)
    diff = np.asarray(errors_b, float) - np.asarray(errors_a, float)
    idx = rng.integers(0, len(diff), size=(draws, len(diff)))
    draws_mean = diff[idx].mean(axis=1)
    low, high = np.percentile(draws_mean, [2.5, 97.5])
    return {
        "delta_MAE": float(diff.mean()),
        "ci95": [float(low), float(high)],
        "distinguishable": bool(low > 0 or high < 0),
    }


def validation_grid(built, cfg, every_days: int, delay: float) -> dict:
    """The same overlap question asked before the test block is opened.

    Both cells fit on the training block and are scored on validation, so the
    comparison is fair; the numbers are not comparable with the test table,
    which calibrates on the whole validation block.
    """
    out = {}
    for with_blocks in (False, True):
        dataset, split = built[with_blocks]
        train, valid = split["train"], split["valid"]
        holdout = max(90, int(cfg.main["quality"].get("conformal_window", 90)))
        model = SulfurModel("rf", 0, [], make_model("rf", cfg.seed), limit=cfg.sulfur_limit)
        model.fit(train.X.iloc[:-holdout], train.y.iloc[:-holdout])
        calibrate_model(model, train.X.iloc[-holdout:], train.y.iloc[-holdout:], cfg)
        point, low, high, _ = model.predict_with_interval(valid.X)
        fixed = {
            "point": point,
            "low": low,
            "high": high,
            "truth": valid.y.to_numpy(),
            "index": valid.y.index,
            "n_fits": 1,
        }
        expanding = expanding_schedule(dataset, {"test": valid}, cfg, every_days, delay)
        label = "с новыми признаками" if with_blocks else "без новых признаков"
        out[label] = {
            "fixed": score(fixed, cfg.sulfur_limit),
            "expanding": score(expanding, cfg.sulfur_limit),
        }
        out[label]["delta_MAE_from_expanding"] = float(
            out[label]["expanding"]["MAE"] - out[label]["fixed"]["MAE"]
        )
    return out


def main(every_days: int = 30) -> dict:
    cfg = load_config()
    interim = project_root() / cfg.main["paths"]["interim"]
    reference = SulfurModel.load(
        project_root() / cfg.main["paths"]["models"] / "sulfur_model.joblib"
    )
    tel = clean_telemetry(load_df(interim / "telemetry.pkl"), cfg)
    lims, pak = load_df(interim / "lims_long.pkl"), load_df(interim / "pak_long.pkl")
    limit, delay = cfg.sulfur_limit, float(cfg.main["data_quality"]["lims_delay_minutes"])

    built = {}
    for with_blocks in (False, True):
        cfg.main["features"].update(controller_activity=with_blocks, catalyst_index=with_blocks)
        dataset = build_dataset_for_model(tel, lims, pak, reference, cfg)
        built[with_blocks] = (dataset, chronological_split(dataset, cfg))

    variants, periods, errors = {}, {}, {}
    for name, spec in GRID.items():
        dataset, split = built[spec["blocks"]]
        result = (
            expanding_schedule(dataset, split, cfg, every_days, delay)
            if spec["expanding"]
            else fixed_schedule(split, cfg)
        )
        variants[name] = {"n_features": int(dataset.X.shape[1]), **score(result, limit)}
        periods[name] = by_period(result, limit, PERIODS)
        errors[name] = np.abs(result["truth"] - result["point"])

    base, final = variants["базовый набор признаков"], variants["переобучение и блоки признаков"]
    alone = (
        variants["расширяющееся переобучение"]["MAE"]
        - base["MAE"]
        + variants["блоки признаков"]["MAE"]
        - base["MAE"]
    )
    report = {
        "test_period": [
            str(built[False][1]["test"].y.index.min()),
            str(built[False][1]["test"].y.index.max()),
        ],
        "every_days": every_days,
        "rejected": {
            "сокращение набора признаков": "выиграл на валидации, проиграл на "
            "блоке оценки, см. reports/feature_selection.json"
        },
        "variants": variants,
        "by_period": periods,
        "validation_check": validation_grid(built, cfg, every_days, delay),
        "significance": {
            f"{name} против базового набора": paired_bootstrap(
                errors["базовый набор признаков"], errors[name], cfg.seed
            )
            for name in GRID
            if name != "базовый набор признаков"
        }
        | {
            "переобучение и блоки против одних блоков": paired_bootstrap(
                errors["блоки признаков"], errors["переобучение и блоки признаков"], cfg.seed
            )
        },
        "additivity": {
            "sum_of_separate_deltas_MAE": float(alone),
            "joint_delta_MAE": float(final["MAE"] - base["MAE"]),
            "note": (
                "если совместная дельта заметно меньше суммы отдельных, приёмы "
                "объясняют одну и ту же часть ошибки"
            ),
        },
    }
    out = project_root() / cfg.main["paths"]["reports"] / "forecast_improvement.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'вариант':36s} {'призн.':>6s} " + " ".join(f"{k:>8s}" for k in REPORTED))
    for name, row in variants.items():
        cells = " ".join(
            f"{row[k]:8.3f}" if isinstance(row[k], float) else f"{str(row[k]):>8s}"
            for k in REPORTED
        )
        print(f"{name:36s} {row['n_features']:6d} {cells}")
    print("\nпо периодам, MAE:")
    print(f"{'вариант':36s} " + " ".join(f"{p:>14s}" for p in PERIODS))
    for name in GRID:
        cells = " ".join(
            f"{periods[name][p]['MAE']:14.3f}" if p in periods[name] else f"{'-':>14s}"
            for p in PERIODS
        )
        print(f"{name:36s} {cells}")
    print("\nпарный бутстрап, разность MAE и её интервал:")
    for label, row in report["significance"].items():
        mark = "отличима" if row["distinguishable"] else "в пределах шума"
        print(
            f"  {label:36s} {row['delta_MAE']:+.3f}  "
            f"[{row['ci95'][0]:+.3f}, {row['ci95'][1]:+.3f}]  {mark}"
        )
    print("\nпроверка на валидации, что даёт переобучение при включённых признаках:")
    for label, row in report["validation_check"].items():
        print(
            f"  {label:24s} MAE {row['fixed']['MAE']:.3f} -> {row['expanding']['MAE']:.3f} "
            f"({row['delta_MAE_from_expanding']:+.3f})"
        )
    print(f"\nсумма отдельных дельт {alone:+.3f}, совместная {final['MAE'] - base['MAE']:+.3f}")
    print(f"-> {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="итоговая сборка улучшений прогноза")
    ap.add_argument("--every-days", type=int, default=30)
    main(**vars(ap.parse_args()))
