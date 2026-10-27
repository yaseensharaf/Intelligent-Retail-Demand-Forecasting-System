"""SARIMA via `pmdarima.auto_arima`. Intervals use `auto_arima`'s own
`.predict(..., return_conf_int=True)` at both alphas -- that's two cheap
re-reads of the already-fitted model, not a second fit.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pmdarima as pm

from forecasting.models.base import INTERVAL_COLUMNS, future_index, normal_intervals

logger = logging.getLogger(__name__)

SEASON_LENGTH = 12


class SARIMAForecaster:
    name = "sarima"

    def __init__(
        self,
        m: int = SEASON_LENGTH,
        max_p: int = 2,
        max_q: int = 2,
        max_P: int = 1,
        max_Q: int = 1,
    ):
        self.m = m
        self.max_p = max_p
        self.max_q = max_q
        self.max_P = max_P
        self.max_Q = max_Q

    def fit(self, y: pd.Series) -> "SARIMAForecaster":
        self.last_date_ = y.index[-1]
        n = len(y)
        seasonal = n >= 2 * self.m

        try:
            self.model_ = pm.auto_arima(
                y.to_numpy(),
                seasonal=seasonal,
                m=self.m if seasonal else 1,
                max_p=self.max_p,
                max_q=self.max_q,
                max_P=self.max_P,
                max_Q=self.max_Q,
                suppress_warnings=True,
                error_action="ignore",
                trace=False,
            )
            self.native_ = True
        except Exception:
            logger.warning("auto_arima failed to fit; falling back to a flat-mean forecast")
            self.native_ = False
            self.mean_ = float(y.mean())
            self.sigma_ = float(y.std(ddof=1) * np.sqrt(1 + 1 / n)) if n > 1 else 0.0
        self.n_ = n
        return self

    def in_sample_residuals(self) -> np.ndarray:
        """In-sample residuals for the hybrid model's step 1. Empty if
        `fit` fell back to the flat-mean model (no SARIMA residuals exist)."""
        if not self.native_:
            return np.zeros(0)
        return np.asarray(self.model_.resid())

    @property
    def difference_order(self) -> int:
        return int(self.model_.order[1]) if self.native_ else 0

    def predict(self, h: int) -> pd.DataFrame:
        if self.native_:
            mean, conf80 = self.model_.predict(n_periods=h, return_conf_int=True, alpha=0.2)
            _, conf95 = self.model_.predict(n_periods=h, return_conf_int=True, alpha=0.05)
            conf80 = np.asarray(conf80, dtype=float)
            conf95 = np.asarray(conf95, dtype=float)
            df = pd.DataFrame({
                "mean": np.asarray(mean, dtype=float),
                "lo80": conf80[:, 0],
                "hi80": conf80[:, 1],
                "lo95": conf95[:, 0],
                "hi95": conf95[:, 1],
            })[INTERVAL_COLUMNS]
        else:
            mean = np.full(h, self.mean_)
            sigma = np.full(h, self.sigma_)
            df = normal_intervals(mean, sigma)
        df.index = future_index(self.last_date_, h)
        return df
