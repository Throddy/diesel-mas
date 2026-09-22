"""Sign-constrained response model, distinct from the quality nowcast.

Coefficients are observational estimates from a closed loop, not causal proof.
No activation energy or undocumented literature prior is inserted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from src.data.operating_mode import operating_modes
from src.features.quality_features import available_feature

TAGS = ["ht:T5", "ht:F9", "ht:P3", "feed_sulfur_mgkg"]


def design(frame):
    """All transformed coefficients are nonnegative by physical sign."""
    return np.column_stack(
        [
            1000 / (frame["ht:T5"] + 273.15),
            np.log(frame["ht:F9"]),
            -np.log(frame["ht:P3"]),
            np.log(frame["feed_sulfur_mgkg"]),
        ]
    )


def _fit(x, y, ridge, prior=None, prior_weight=0.0, constrained=True):
    """Ridge fit; ``constrained`` keeps the physical signs, free lets data decide."""
    matrix = np.column_stack([np.ones(len(x)), x])
    penalty = np.diag([0.0, *([ridge**0.5] * x.shape[1])])
    target = np.zeros(x.shape[1] + 1)
    if prior is not None:
        penalty[1, 1] = prior_weight**0.5
        target[1] = penalty[1, 1] * prior
    lower = np.r_[-np.inf, np.zeros(x.shape[1])] if constrained else -np.inf
    return lsq_linear(np.vstack([matrix, penalty]), np.r_[y, target], bounds=(lower, np.inf)).x


@dataclass
class ResponseModel:
    center: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    bootstrap: np.ndarray
    bounds: dict
    lag_minutes: int
    diagnostics: dict

    @classmethod
    def fit(cls, tel, target, feed, cfg, constrained: bool = True, use_prior: bool = True):
        """Fit M, or the unconstrained variant used by architecture E.

        With ``constrained=False`` and ``use_prior=False`` the same features are
        regressed with no sign bounds and no physical prior: the straightforward
        fit, which on this closed loop puts the wrong sign on temperature.
        """
        response = cfg.main["response"]
        dq = cfg.main["data_quality"]
        end = pd.Timestamp(cfg.main["split"]["train_end"])
        y = target.loc[:end]
        y = y[y.between(0, cfg.main["quality"]["max_plausible_sulfur_mgkg"], inclusive="right")]
        at = y.index - pd.Timedelta(minutes=response["lag_minutes"])
        x = tel[TAGS[:3]].reindex(at, method="ffill").set_axis(y.index)
        ready = available_feature(
            at, feed, "feed_sulfur_mgkg", dq["lims_delay_minutes"], dq["feed_lims_max_age_minutes"]
        )
        x["feed_sulfur_mgkg"] = ready["feed_sulfur_mgkg"].to_numpy()
        eligible = operating_modes(tel, cfg)["eligible"]
        eligible = eligible.reindex(at, method="ffill").fillna(False).to_numpy()
        keep = eligible & x.notna().all(axis=1) & (x > 0).all(axis=1)
        x, y = x.loc[keep], y.loc[keep]
        if len(x) < response["min_samples"]:
            raise ValueError(f"Insufficient response samples: {len(x)}")

        matrix = design(x)
        center = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        if np.any(scale == 0):
            raise ValueError("Constant response regressor")
        z = (matrix - center) / scale
        log_y = np.log(y.to_numpy())

        sign_fit = _fit(z, log_y, response["ridge"])
        prior = None
        if response["prior_enabled"] and use_prior:
            prior = (
                response["activation_energy_kj_mol"] / response["gas_constant_j_mol_k"] * scale[0]
            )
        prior_weight = len(x) * response["prior_weight_per_sample"]
        coefficient = _fit(z, log_y, response["ridge"], prior, prior_weight, constrained)

        rng = np.random.default_rng(cfg.seed)
        weeks = x.index.to_period("W")
        unique_weeks = weeks.unique()
        spread = response["activation_energy_spread_kj_mol"]
        centre_energy = response["activation_energy_kj_mol"]
        bootstrap = []
        for _ in range(response["bootstrap_blocks"]):
            drawn = rng.choice(unique_weeks, len(unique_weeks), replace=True)
            index = np.concatenate([np.flatnonzero(weeks == week) for week in drawn])
            energy = rng.uniform(centre_energy - spread, centre_energy + spread)
            prior_draw = None
            if prior is not None:
                prior_draw = energy / response["gas_constant_j_mol_k"] * scale[0]
            bootstrap.append(
                _fit(
                    z[index], log_y[index], response["ridge"], prior_draw, prior_weight, constrained
                )
            )

        design_matrix = np.column_stack([np.ones(len(z)), z])
        unconstrained = np.linalg.lstsq(design_matrix, log_y, rcond=None)[0]
        median = x.median()
        per_degree = -1000 / (median["ht:T5"] + 273.15) ** 2 / scale[0]
        quantiles = np.quantile(np.array(bootstrap)[:, 1] * per_degree, [0.05, 0.95])
        diagnostics = {
            "n_train": len(x),
            "train_end": str(end),
            "lag_minutes": response["lag_minutes"],
            "lag_basis": "A-18: заданное модельное запаздывание, не оценка времени " "пребывания",
            "temperature_log_sensitivity_data": float(unconstrained[1] * per_degree),
            "constrained": bool(constrained),
            "prior_used": bool(prior is not None),
            "temperature_log_sensitivity_sign_constrained": float(sign_fit[1] * per_degree),
            "temperature_log_sensitivity_prior": (
                float(prior * per_degree) if prior is not None else None
            ),
            "temperature_log_sensitivity_joint": float(coefficient[1] * per_degree),
            "prior_status": "A-19: литературное E_a перенесено в приближённую "
            "log-модель; это допущение, не параметр данной установки",
            "prior_source": response["prior_source"],
            "temperature_sensitivity_bootstrap": quantiles.tolist(),
            "coefficients": dict(zip(TAGS, (coefficient[1:] / scale).tolist())),
            "limitation": "Замкнутый контур, редкая сера сырья; причинность и перенос "
            "на установку не доказаны",
        }
        bounds = {tag: [float(x[tag].min()), float(x[tag].max())] for tag in TAGS}
        return cls(
            center,
            scale,
            coefficient,
            np.array(bootstrap),
            bounds,
            response["lag_minutes"],
            diagnostics,
        )

    def effect(self, before: dict, after: dict):
        """Return central and pessimistic log changes, within training support."""
        frame = pd.DataFrame([before, after])[TAGS]
        if frame.isna().any().any() or (frame <= 0).any().any():
            raise ValueError("Response inputs missing or nonpositive")
        for tag, (low, high) in self.bounds.items():
            if not frame[tag].between(low, high).all():
                raise ValueError(f"Response outside support: {tag}")
        d = (design(frame)[1] - design(frame)[0]) / self.scale
        return float(d @ self.coefficient[1:]), float(np.quantile(self.bootstrap[:, 1:] @ d, 0.9))

    def save(self, path):
        import joblib

        joblib.dump(self, path)

    @staticmethod
    def load(path):
        import joblib

        return joblib.load(path)
