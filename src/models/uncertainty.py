"""Split-conformal uncertainty with availability-aware residual selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

STATUS_OK = "OK"
STATUS_INSUFFICIENT = "INSUFFICIENT_HISTORY"
STATUS_NO_TIMESTAMPS = "NO_AVAILABILITY_TIMES"


@dataclass
class ConformalCalibration:
    """Conformal quantiles from residuals reported before the decision moment.

    ``timestamps`` hold the moment each residual became available, which for a
    laboratory residual is the sampling moment plus the reporting delay.
    ``as_of`` selects residuals strictly earlier than the requested moment
    regardless of ``window`` and carries the timestamps into the derived object,
    so a repeated call stays availability-aware. A selection that leaves fewer
    than ``minimum`` residuals yields an undefined quantile and the status
    ``INSUFFICIENT_HISTORY``; residuals from the future are never substituted.
    Without timestamps availability cannot be established, so ``as_of`` returns
    an empty calibration with the status ``NO_AVAILABILITY_TIMES``.
    """

    residuals: np.ndarray
    alpha: float = 0.1
    timestamps: Optional[np.ndarray] = None
    window: int = 0
    minimum: int = 10
    status: str = field(default=STATUS_OK)

    def __post_init__(self) -> None:
        self.residuals = np.asarray(self.residuals, dtype=float)
        if self.timestamps is not None:
            self.timestamps = np.asarray(self.timestamps)
        if self.status == STATUS_OK and len(self.residuals) < self.minimum:
            self.status = STATUS_INSUFFICIENT

    @property
    def n(self) -> int:
        return int(len(self.residuals))

    @property
    def usable(self) -> bool:
        return self.status == STATUS_OK and self.n >= self.minimum

    def as_of(self, moment) -> "ConformalCalibration":
        """Residuals reported before ``moment``, at most ``window`` of them."""
        if self.timestamps is None:
            return ConformalCalibration(
                residuals=np.empty(0, dtype=float),
                alpha=self.alpha,
                timestamps=np.empty(0, dtype="datetime64[ns]"),
                window=self.window,
                minimum=self.minimum,
                status=STATUS_NO_TIMESTAMPS,
            )
        cutoff = _as_datetime64(moment)
        if cutoff is None:
            selected = np.zeros(len(self.residuals), dtype=bool)
        else:
            selected = np.asarray(self.timestamps) < cutoff
        residuals = self.residuals[selected]
        timestamps = self.timestamps[selected]
        if self.window:
            residuals = residuals[-self.window :]
            timestamps = timestamps[-self.window :]
        return ConformalCalibration(
            residuals=residuals,
            alpha=self.alpha,
            timestamps=timestamps,
            window=self.window,
            minimum=self.minimum,
        )

    def quantile(self, alpha: Optional[float] = None) -> float:
        return self._quantile(np.abs(self.residuals), alpha)

    def upper_quantile(self, alpha: Optional[float] = None) -> float:
        """One-sided upper quantile of the signed residuals for the safety check."""
        return self._quantile(self.residuals, alpha)

    def _quantile(self, values: np.ndarray, alpha: Optional[float]) -> float:
        if not self.usable:
            return float("nan")
        level = min(
            1.0, np.ceil((self.n + 1) * (1 - (self.alpha if alpha is None else alpha))) / self.n
        )
        return float(np.quantile(values, level, method="higher"))

    def upper_log(self, pred_log: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
        return np.asarray(pred_log, dtype=float) + self.upper_quantile(alpha)

    def interval_log(
        self, pred_log: np.ndarray, alpha: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        spread = self.quantile(alpha)
        return pred_log - spread, pred_log + spread

    def exceedance_risk(self, pred_log: np.ndarray, limit: float) -> np.ndarray:
        """Share of available residuals that push the prediction above ``limit``."""
        if not self.usable:
            return np.full(np.shape(pred_log), np.nan)
        threshold = np.log(limit) - np.asarray(pred_log, dtype=float)[..., None]
        return (self.residuals[None, :] > threshold).mean(axis=-1)


def _as_datetime64(moment) -> Optional[np.datetime64]:
    try:
        return np.datetime64(moment)
    except (ValueError, TypeError):
        return None


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))
