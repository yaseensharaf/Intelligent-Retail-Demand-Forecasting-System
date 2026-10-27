"""ETS (error-trend-seasonal) exponential smoothing, via statsmodels'
state-space `ETSModel` -- unlike `ExponentialSmoothing`, it has a native
analytical prediction interval. Only additive error/trend/seasonal
components are used: `data.to_monthly` zero-fills calendar gaps, and a
multiplicative component on a zero month is a real convergence risk given
how many Store 2 products are intermittent, not an edge case.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from forecasting.models.base import INTERVAL_COLUMNS, future_index, normal_intervals

logger = logging.getLogger(__name__)

SEASON_LENGTH = 12


class ETSForecaster:
    name = "ets"

    def __init__(self, seasonal_periods: int = SEASON_LENGTH):
        self.seasonal_periods = seasonal_periods

    def fit(self, y: pd.Series) -> "ETSForecaster":
        self.last_date_ = y.index[-1]
        self.n_ = len(y)
        has_seasonal = self.n_ >= 2 * self.seasonal_periods

        try:
            model = ETSModel(
                y,
                error="add",
                trend="add",
                seasonal="add" if has_seasonal else None,
                seasonal_periods=self.seasonal_periods if has_seasonal else None,
            )
            self.results_ = model.fit(disp=False)
            self.native_ = True
        except Exception:
            logger.warning("ETS fit failed to converge; falling back to a flat-level forecast")
            self.native_ = False
            self.mean_ = float(y.mean())
            self.sigma_ = float(y.std(ddof=1) * np.sqrt(1 + 1 / self.n_)) if self.n_ > 1 else 0.0
        return self

    def predict(self, h: int) -> pd.DataFrame:
        if self.native_:
            pred = self.results_.get_prediction(start=self.n_, end=self.n_ + h - 1, method="exact")
            summary80 = pred.summary_frame(alpha=0.2)
            summary95 = pred.summary_frame(alpha=0.05)
            df = pd.DataFrame({
                "mean": summary80["mean"].to_numpy(),
                "lo80": summary80["pi_lower"].to_numpy(),
                "hi80": summary80["pi_upper"].to_numpy(),
                "lo95": summary95["pi_lower"].to_numpy(),
                "hi95": summary95["pi_upper"].to_numpy(),
            })[INTERVAL_COLUMNS]
        else:
            mean = np.full(h, self.mean_)
            sigma = np.full(h, self.sigma_)
            df = normal_intervals(mean, sigma)
        df.index = future_index(self.last_date_, h)
        return df
