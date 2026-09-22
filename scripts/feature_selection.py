"""Which of the current features earn their place?

The prototype had 645 features on 850 analyses; that is long fixed, and the
model now uses 50.  The question is no longer whether to cut hundreds, but
whether these fifty carry independent information or repeat each other.

Selection is computed inside the training block with rolling-origin folds, so
neither validation nor test influences which features survive.  The test block
is opened once, at the end, to report the outcome.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.config import load_config, project_root
from src.data.loaders import clean_telemetry
from src.data.store import load_df
from src.features.dataset import build_dataset_for_model, chronological_split
from src.models.quality import SulfurModel, calibrate_model, make_model, regression_metrics


def rolling_importance(
    X: pd.DataFrame, y: pd.Series, cfg, n_folds: int = 3, repeats: int = 3
) -> pd.DataFrame:
    """Permutation importance averaged over rolling origins inside training.

    Each fold trains on a prefix and scores the block that follows it, which
    is how the model will be used.  Shuffling one column at a time measures
    what the model loses without it.
    """
    rng = np.random.default_rng(cfg.seed)
    scores = {name: [] for name in X.columns}
    cuts = [int(len(y) * f) for f in np.linspace(0.5, 0.85, n_folds)]
    for cut in cuts:
        stop = min(len(y), cut + max(30, len(y) // 6))
        fit_X, fit_y = X.iloc[:cut], y.iloc[:cut]
        test_X, test_y = X.iloc[cut:stop], y.iloc[cut:stop]
        if len(test_y) < 10:
            continue
        model = SulfurModel("rf", 0, [], make_model("rf", cfg.seed)).fit(fit_X, fit_y)
        base = regression_metrics(test_y.to_numpy(), model.predict(test_X))["MAE_log"]
        for name in X.columns:
            losses = []
            for _ in range(repeats):
                shuffled = test_X.copy()
                shuffled[name] = rng.permutation(shuffled[name].to_numpy())
                losses.append(
                    regression_metrics(test_y.to_numpy(), model.predict(shuffled))["MAE_log"] - base
                )
            scores[name].append(float(np.mean(losses)))
    return (
        pd.DataFrame(
            {
                "feature": list(scores),
                "importance": [float(np.mean(v)) if v else 0.0 for v in scores.values()],
                "folds": [len(v) for v in scores.values()],
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def drop_duplicates(X: pd.DataFrame, ranked: pd.DataFrame, threshold: float) -> list[str]:
    """Keep the higher-ranked feature of any near-duplicate pair."""
    corr = X.corr().abs()
    keep: list[str] = []
    for name in ranked.feature:
        if any(corr.loc[name, kept] > threshold for kept in keep):
            continue
        keep.append(name)
    return keep


def evaluate(features: list[str], split, cfg) -> dict:
    """Fit on training with the given features and score the validation block."""
    model = SulfurModel("rf", 0, [], make_model("rf", cfg.seed), limit=cfg.sulfur_limit).fit(
        split["train"].X[features], split["train"].y
    )
    valid = split["valid"]
    return regression_metrics(valid.y.to_numpy(), model.predict(valid.X[features]))


def main(correlation_threshold: float = 0.95, open_test: bool = True) -> dict:
    cfg = load_config()
    interim = project_root() / cfg.main["paths"]["interim"]
    model = SulfurModel.load(project_root() / cfg.main["paths"]["models"] / "sulfur_model.joblib")
    tel = clean_telemetry(load_df(interim / "telemetry.pkl"), cfg)
    lims, pak = load_df(interim / "lims_long.pkl"), load_df(interim / "pak_long.pkl")
    dataset = build_dataset_for_model(tel, lims, pak, model, cfg)
    split = chronological_split(dataset, cfg)
    train = split["train"]

    ranked = rolling_importance(train.X, train.y, cfg)
    useful = ranked[ranked.importance > 0]
    kept = drop_duplicates(train.X, useful, correlation_threshold)

    variants = {
        "все признаки": list(train.X.columns),
        "с ненулевым вкладом": list(useful.feature),
        "без дубликатов": kept,
        "20 лучших": list(ranked.feature.head(20)),
        "10 лучших": list(ranked.feature.head(10)),
    }
    on_validation = {
        name: evaluate(features, split, cfg) for name, features in variants.items() if features
    }

    best_name = min(on_validation, key=lambda k: on_validation[k]["MAE_log"])
    report = {
        "n_features_total": int(train.X.shape[1]),
        "n_training_rows": int(len(train.y)),
        "method": (
            "перестановочная важность на трёх скользящих отсечениях внутри "
            "обучающего блока, затем отсев признаков с корреляцией выше "
            f"{correlation_threshold}"
        ),
        "correlation_threshold": correlation_threshold,
        "ranking": ranked.to_dict("records"),
        "variants": {name: {"n": len(features)} for name, features in variants.items()},
        "validation": on_validation,
        "best_on_validation": best_name,
        "selected_features": variants[best_name],
    }

    if open_test and best_name != "все признаки":
        for name in ("все признаки", best_name):
            features = variants[name]
            fitted = SulfurModel(
                "rf", 0, [], make_model("rf", cfg.seed), limit=cfg.sulfur_limit
            ).fit(train.X[features], train.y)
            calibrate_model(fitted, split["valid"].X[features], split["valid"].y, cfg)
            point, low, high, _ = fitted.predict_with_interval(split["test"].X[features])
            truth = split["test"].y.to_numpy()
            report.setdefault("test", {})[name] = {
                **regression_metrics(truth, point),
                "coverage": float(np.mean((truth >= low) & (truth <= high))),
                "mean_width": float(np.mean(high - low)),
            }

    out = project_root() / cfg.main["paths"]["reports"] / "feature_selection.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "validation": on_validation,
                "best": best_name,
                "variants": report["variants"],
                "test": report.get("test"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\n-> {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="отбор признаков внутри обучающего блока")
    ap.add_argument("--correlation-threshold", type=float, default=0.95)
    main(correlation_threshold=ap.parse_args().correlation_threshold)
