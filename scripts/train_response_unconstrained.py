"""Fit the response model the straightforward way, for architecture E.

Same data, same feature form and same ridge as M, with two things removed: the
nonnegativity bounds that encode the physical signs, and the activation-energy
prior.  This is what an ordinary regression on this plant produces.

The result is saved beside M and is never loaded by the shipped pipeline; only
`scripts/arch_compare.py` and `scripts/simulate.py` pick it up, and only for
configuration E.
"""

from __future__ import annotations

import argparse
import json

from src.config import load_config, project_root
from src.data.loaders import clean_telemetry, feed_sulfur, target_series
from src.data.store import load_df
from src.models.response_model import TAGS, ResponseModel

ARTIFACT = "response_model_unconstrained.joblib"

EXPECTED_NONNEGATIVE = dict(
    zip(
        TAGS,
        (
            "рост температуры снижает серу",
            "рост расхода сырья повышает серу",
            "рост давления водорода снижает серу",
            "рост серы сырья повышает серу продукта",
        ),
        strict=False,
    )
)


def main() -> dict:
    cfg = load_config()
    interim = project_root() / cfg.main["paths"]["interim"]
    models = project_root() / cfg.main["paths"]["models"]
    tel = clean_telemetry(load_df(interim / "telemetry.pkl"), cfg)
    lims = load_df(interim / "lims_long.pkl")
    target, feed = target_series(lims, cfg), feed_sulfur(lims)

    free = ResponseModel.fit(tel, target, feed, cfg, constrained=False, use_prior=False)
    free.save(models / ARTIFACT)

    scaled = free.coefficient[1:] / free.scale
    signs = {
        tag: ("соответствует физике" if value >= 0 else "противоречит физике")
        for tag, value in zip(TAGS, scaled, strict=False)
    }
    reference = ResponseModel.load(models / "response_model.joblib")
    report = {
        "artifact": f"models/{ARTIFACT}",
        "purpose": "конфигурация E в сравнении архитектур; в рабочий конвейер не подключается",
        "difference_from_M": "сняты границы неотрицательности коэффициентов и априорный центр",
        "n_train": free.diagnostics["n_train"],
        "train_end": free.diagnostics["train_end"],
        "coefficients": {t: float(v) for t, v in zip(TAGS, scaled, strict=False)},
        "expected_sign": dict.fromkeys(TAGS, "неотрицательный"),
        "sign_verdict": signs,
        "n_signs_against_physics": int(sum(v < 0 for v in scaled)),
        "temperature_log_sensitivity": {
            "unconstrained_E": free.diagnostics["temperature_log_sensitivity_joint"],
            "joint_M": reference.diagnostics["temperature_log_sensitivity_joint"],
            "prior_M": reference.diagnostics["temperature_log_sensitivity_prior"],
        },
        "meaning": EXPECTED_NONNEGATIVE,
    }
    out = project_root() / cfg.main["paths"]["reports"] / "response_unconstrained.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'параметр':22s} {'коэффициент':>14s}  вердикт")
    for tag, value in zip(TAGS, scaled, strict=False):
        print(f"{tag:22s} {value:14.6f}  {signs[tag]}")
    s = report["temperature_log_sensitivity"]
    print(
        f"\nd ln S / dT: E {s['unconstrained_E']:+.4f}, M {s['joint_M']:+.4f}, "
        f"априори {s['prior_M']:+.4f} 1/°C"
    )
    print(f"знаков против физики: {report['n_signs_against_physics']} из {len(TAGS)}")
    print(f"-> {out}")
    return report


if __name__ == "__main__":
    argparse.ArgumentParser(description="модель отклика без ограничений знаков").parse_args()
    main()
