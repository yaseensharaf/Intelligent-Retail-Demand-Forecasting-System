"""Shared forecaster contract. `future_index` is the single place forecast
dates are created, so every model agrees on where "step 1" falls."""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

INTERVAL_COLUMNS = ["mean", "lo80", "hi80", "lo95", "hi95"]

Z80 = 1.2815515655446004
Z95 = 1.959963984540054


class Forecaster(Protocol):
    name: str

    def fit(self, y: pd.Series) -> "Forecaster": ...

    def predict(self, h: int) -> pd.DataFrame: ...


def future_index(last_date: pd.Timestamp, h: int) -> pd.DatetimeIndex:
    start = last_date + pd.offsets.MonthEnd(1)
    return pd.date_range(start, periods=h, freq="ME")


def normal_intervals(mean: np.ndarray, sigma: np.ndarray) -> pd.DataFrame:
    mean = np.asarray(mean, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return pd.DataFrame({
        "mean": mean,
        "lo80": mean - Z80 * sigma,
        "hi80": mean + Z80 * sigma,
        "lo95": mean - Z95 * sigma,
        "hi95": mean + Z95 * sigma,
    })[INTERVAL_COLUMNS]
