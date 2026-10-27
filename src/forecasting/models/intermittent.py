"""Croston's method and its TSB (Teunter-Syntetos-Babai) variant, for
intermittent/sparse demand. Neither is covered by a pinned package, so
both are hand-implemented here.

Both produce a flat level repeated for every step of the forecast, so --
like `HistoricalMeanForecaster` -- the prediction interval is a single
scalar sigma, not one that grows with the horizon. Neither method has a
closed-form parametric interval; sigma is approximated as the std of
one-step-ahead fitted errors over the full, zero-inclusive train series.
`lo80`/`lo95` are clipped at 0 since quantity sold can't be negative --
a real possibility once these models target genuinely sparse products,
not a theoretical edge case.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.models.base import future_index, normal_intervals

DEFAULT_ALPHA = 0.1
DEFAULT_BETA = 0.1


def _croston_fit(values: np.ndarray, alpha: float) -> tuple[float, np.ndarray]:
    fitted = np.zeros(len(values))
    nonzero = np.flatnonzero(values > 0)
    if len(nonzero) == 0:
        return 0.0, fitted

    idx0 = int(nonzero[0])
    z_hat = float(values[idx0])
    p_hat = float(idx0 + 1)
    level = z_hat / p_hat
    since_last = 0

    for t in range(idx0, len(values)):
        fitted[t] = level
        since_last += 1
        if t == idx0:
            since_last = 0
            continue
        if values[t] > 0:
            z_hat += alpha * (values[t] - z_hat)
            p_hat += alpha * (since_last - p_hat)
            p_hat = max(p_hat, 1e-6)
            since_last = 0
            level = z_hat / p_hat
    return level, fitted


def _tsb_fit(values: np.ndarray, alpha: float, beta: float) -> tuple[float, np.ndarray]:
    fitted = np.zeros(len(values))
    nonzero = np.flatnonzero(values > 0)
    if len(nonzero) == 0:
        return 0.0, fitted

    idx0 = int(nonzero[0])
    z_hat = float(values[idx0])
    p_hat = 1.0 / float(idx0 + 1)
    level = p_hat * z_hat

    for t in range(idx0, len(values)):
        fitted[t] = level
        if t != idx0:
            demand_occurred = values[t] > 0
            p_hat += beta * (float(demand_occurred) - p_hat)
            if demand_occurred:
                z_hat += alpha * (values[t] - z_hat)
            level = p_hat * z_hat
    return level, fitted


def _clip_lower_zero(df: pd.DataFrame) -> pd.DataFrame:
    df["lo80"] = df["lo80"].clip(lower=0.0)
    df["lo95"] = df["lo95"].clip(lower=0.0)
    return df


class CrostonForecaster:
    name = "croston"

    def __init__(self, alpha: float = DEFAULT_ALPHA):
        self.alpha = alpha

    def fit(self, y: pd.Series) -> "CrostonForecaster":
        self.last_date_ = y.index[-1]
        values = y.to_numpy()
        self.level_, fitted = _croston_fit(values, self.alpha)
        resid = values - fitted
        self.sigma_ = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
        return self

    def predict(self, h: int) -> pd.DataFrame:
        mean = np.full(h, self.level_)
        sigma = np.full(h, self.sigma_)
        df = _clip_lower_zero(normal_intervals(mean, sigma))
        df.index = future_index(self.last_date_, h)
        return df


class TSBForecaster:
    name = "tsb"

    def __init__(self, alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA):
        self.alpha = alpha
        self.beta = beta

    def fit(self, y: pd.Series) -> "TSBForecaster":
        self.last_date_ = y.index[-1]
        values = y.to_numpy()
        self.level_, fitted = _tsb_fit(values, self.alpha, self.beta)
        resid = values - fitted
        self.sigma_ = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
        return self

    def predict(self, h: int) -> pd.DataFrame:
        mean = np.full(h, self.level_)
        sigma = np.full(h, self.sigma_)
        df = _clip_lower_zero(normal_intervals(mean, sigma))
        df.index = future_index(self.last_date_, h)
        return df
