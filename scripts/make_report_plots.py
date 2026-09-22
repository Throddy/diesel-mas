"""EDA and model-report figures (matplotlib, no interactive backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config, project_root
from src.data.loaders import clean_telemetry, pak_d15, pak_sulfur, split_bounds, target_series
from src.data.store import load_df
from src.data.validation import find_flatlines
from src.features.dataset import build_dataset_for_model, chronological_split
from src.models.quality import SulfurModel


def store(fig, fig_dir: Path, name: str, saved: list) -> None:
    """Write one figure and record its name."""
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=120)
    plt.close(fig)
    saved.append(name)


def main() -> None:
    cfg = load_config()
    fig_dir = project_root() / cfg.main["paths"]["figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    interim = project_root() / cfg.main["paths"]["interim"]
    tel = clean_telemetry(load_df(interim / "telemetry.pkl"), cfg)
    lims, pak = load_df(interim / "lims_long.pkl"), load_df(interim / "pak_long.pkl")
    y, ps, pdn = target_series(lims, cfg), pak_sulfur(pak), pak_d15(pak)
    limit = cfg.sulfur_limit
    saved = []

    fig, ax = plt.subplots(figsize=(11, 3.2))
    for i, (_name, idx) in enumerate(
        [
            ("телеметрия", tel.index),
            ("ПАК сера", ps.index),
            ("ПАК D15", pdn.index),
            ("ЛИМС сера", y.index),
        ]
    ):
        ax.plot(idx, np.full(len(idx), i), "|", markersize=6)
    ax.set_yticks(range(4), ["телеметрия", "ПАК сера", "ПАК D15", "ЛИМС сера"])
    ax.set_title("Покрытие источников во времени")
    for t, ls in zip(split_bounds(cfg), ["--", "--", "-"], strict=False):
        ax.axvline(t, color="k", ls=ls, lw=0.8)
    store(fig, fig_dir, "01_coverage.png", saved)

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(ps.index, ps.values, lw=0.4, label="ПАК")
    ax.plot(y.index, np.clip(y.values, 0, 40), ".", ms=3, label="ЛИМС (обрезано 40)")
    ax.axhline(limit, color="r", ls="--", label="предел 10 мг/кг")
    for seg in find_flatlines(ps, int(cfg.dq("flatline_critical_points"))):
        ax.axvspan(seg.start, seg.end, color="orange", alpha=0.3)
    ax.legend()
    ax.set_title("Сера, ПАК против ЛИМС; оранжевым отмечен флэтлайн анализатора")
    store(fig, fig_dir, "02_pak_vs_lims.png", saved)

    model = SulfurModel.load(project_root() / cfg.main["paths"]["models"] / "sulfur_model.joblib")
    ds = build_dataset_for_model(tel, lims, pak, model, cfg)
    sp = chronological_split(ds, cfg)
    blk = sp["test"]
    p, lo, hi, risk = model.predict_with_interval(blk.X)

    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(blk.y, p, s=10, alpha=0.6)
    m = max(float(np.nanmax(p)), 30)
    ax.plot([0, m], [0, m], "k--", lw=0.8)
    ax.axhline(limit, color="r", lw=0.7)
    ax.axvline(limit, color="r", lw=0.7)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_xlabel("ЛИМС, мг/кг")
    ax.set_ylabel("прогноз, мг/кг")
    ax.set_title("Блок оценки, факт против прогноза")
    store(fig, fig_dir, "03_actual_vs_pred.png", saved)

    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(blk.y.index, blk.y.values, ".", ms=4, label="ЛИМС")
    ax.plot(blk.y.index, p, lw=1, label="прогноз")
    ax.fill_between(blk.y.index, lo, hi, alpha=0.25, label="интервал (номинально 90 %)")
    ax.axhline(limit, color="r", ls="--")
    ax.set_ylim(0, 30)
    ax.legend()
    ax.set_title("Блок оценки, прогноз с интервалом и предел")
    store(fig, fig_dir, "04_forecast_interval.png", saved)

    imp = load_df(project_root() / cfg.main["paths"]["models"] / "feature_importance.pkl").head(15)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp["feature"][::-1], imp["importance"][::-1])
    ax.set_title("Permutation importance (validation, log-MAE)")
    store(fig, fig_dir, "05_feature_importance.png", saved)

    from src.models.reliability import ReliabilityModel

    rel = ReliabilityModel.load(
        project_root() / cfg.main["paths"]["models"] / "reliability_model.joblib"
    )
    sev = rel.severity_frame(tel.iloc[::36])
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(sev.index, sev["severity_index"], lw=0.5)
    ax.axhline(rel.severity_warn, color="orange", ls="--", label="ELEVATED")
    ax.axhline(rel.severity_high, color="r", ls="--", label="HIGH")
    ax.legend()
    ax.set_title("История индекса тяжести режима, прокси")
    store(fig, fig_dir, "06_severity.png", saved)

    print("saved:", ", ".join(saved))


if __name__ == "__main__":
    main()
