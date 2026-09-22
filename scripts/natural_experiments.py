"""Measure the effect of control moves on episodes the operators created.

Model M gets its magnitude from data plus a physical prior.  This checks the
prior against the unit itself: episodes where a control was stepped and held,
with nothing else moving, and the sulfur that followed.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.config import load_config, project_root
from src.data.loaders import clean_telemetry, feed_sulfur, pak_sulfur, target_series
from src.data.operating_mode import operating_modes
from src.data.store import load_df
from src.eval.natural_experiments import (
    FEED,
    TEMPERATURE,
    attach_sulfur,
    criterion_map,
    find_episodes,
    sensitivity,
    usable,
)
from src.models.response_model import ResponseModel


def predicted_change(model: ResponseModel, episode, state: dict) -> float | None:
    """What model M expects for this step, at the episode's own operating point."""
    before = dict(state)
    after = dict(state)
    after[episode.control] = episode.value_after
    before[episode.control] = episode.value_before
    try:
        central, _ = model.effect(before, after)
    except ValueError:
        return None
    return float(central)


def main(min_episodes: int | None = None) -> dict:
    cfg = load_config()
    section = cfg.main["natural_experiments"]
    interim = project_root() / cfg.main["paths"]["interim"]
    tel = clean_telemetry(load_df(interim / "telemetry.pkl"), cfg)
    lims = load_df(interim / "lims_long.pkl")
    pak = load_df(interim / "pak_long.pkl")
    eligible = operating_modes(tel, cfg)["eligible"]

    laboratory = target_series(lims, cfg)
    laboratory = laboratory[(laboratory > 0) & (laboratory <= 100)]
    analyser = pak_sulfur(pak)
    feed = feed_sulfur(lims)
    lag = float(cfg.main["response"]["lag_minutes"])
    model = ResponseModel.load(
        project_root() / cfg.main["paths"]["models"] / "response_model.joblib"
    )
    diagnostics = json.loads(
        (project_root() / cfg.main["paths"]["models"] / "response_model.json").read_text(
            encoding="utf-8"
        )
    )

    report = {
        "thresholds": {k: v for k, v in section.items() if k != "basis"},
        "basis": section["basis"],
        "controls": {},
    }

    for control in (TEMPERATURE, FEED):
        episodes = find_episodes(tel, eligible, control, section)
        episodes = attach_sulfur(episodes, laboratory, analyser, feed, section, lag)
        good = usable(episodes)

        rows = []
        for episode in good:
            state = tel.loc[: episode.change_time].iloc[-1].to_dict()
            state["feed_sulfur_mgkg"] = episode.feed_sulfur_before
            if not state.get("feed_sulfur_mgkg"):
                continue
            expected = predicted_change(model, episode, state)
            row = episode.to_dict()
            row["predicted_log_change"] = expected
            rows.append(row)

        observed = np.array(
            [r["observed_log_change"] for r in rows if r["predicted_log_change"] is not None]
        )
        expected = np.array(
            [r["predicted_log_change"] for r in rows if r["predicted_log_change"] is not None]
        )
        paired = {
            "n_paired": int(len(observed)),
            "sign_agreement": (
                float(np.mean(np.sign(observed) == np.sign(expected))) if len(observed) else None
            ),
            "correlation": (
                float(np.corrcoef(observed, expected)[0, 1])
                if len(observed) > 2 and np.std(expected) > 0
                else None
            ),
        }
        report["controls"][control] = {
            "episodes_found": len(episodes),
            "episodes_with_sulfur": len(good),
            "sources": {
                source: sum(1 for e in good if e.sulfur_source == source)
                for source in ("LIMS", "PAK")
            },
            "excluded_feed_change": sum(
                1 for e in episodes if any("исключён" in n for n in e.notes)
            ),
            "sensitivity_per_unit": sensitivity(good),
            "paired_with_model_M": paired,
            "criterion_map": criterion_map(tel, eligible, control, section),
            "episodes": rows,
        }

    threshold = (
        min_episodes if min_episodes is not None else int(section["min_episodes_for_conclusion"])
    )
    temperature = report["controls"][TEMPERATURE]
    n = temperature["sensitivity_per_unit"].get("n", 0)
    report["comparison_d_ln_s_dt"] = {
        "natural_experiments": temperature["sensitivity_per_unit"].get("median"),
        "natural_experiments_ci90": temperature["sensitivity_per_unit"].get("ci90"),
        "model_M_data_only": diagnostics["temperature_log_sensitivity_data"],
        "model_M_prior": diagnostics["temperature_log_sensitivity_prior"],
        "model_M_joint": diagnostics["temperature_log_sensitivity_joint"],
        "plant_model": -0.0659,
    }
    report["conclusion"] = {
        "episodes": n,
        "threshold": threshold,
        "sufficient": bool(n >= threshold),
        "statement": (
            "эпизодов достаточно для оценки"
            if n >= threshold
            else f"эпизодов {n}, меньше порога {threshold}: вывод о величине эффекта не строится"
        ),
        "why_none": (
            "при скачке температуры размах внутри 12-часового окна имеет медиану "
            "9,2 °C: регулятор не оставляет ступенек с удержанием. Удержание короче "
            "транспортного запаздывания 6 ч физически непригодно, а при 12 ч эпизодов "
            "нет ни при одном допуске, см. criterion_map"
        ),
    }
    out = project_root() / cfg.main["paths"]["reports"] / "natural_experiments.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    summary = {
        control: {
            k: data[k]
            for k in (
                "episodes_found",
                "episodes_with_sulfur",
                "sensitivity_per_unit",
                "paired_with_model_M",
            )
        }
        for control, data in report["controls"].items()
    }
    print(
        json.dumps(
            {
                "summary": summary,
                "comparison": report["comparison_d_ln_s_dt"],
                "conclusion": report["conclusion"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\n-> {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="натуральные эксперименты по историческим данным")
    ap.add_argument("--min-episodes", type=int, default=None)
    a = ap.parse_args()
    main(a.min_episodes)
