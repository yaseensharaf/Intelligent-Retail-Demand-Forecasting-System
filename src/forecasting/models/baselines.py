"""Baseline forecasters. Forecast formulas and interval-width formulas follow
Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed.,
sections 5.2 (simple methods) and 5.4 (prediction intervals).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.models.base import future_index, normal_intervals


class NaiveForecaster:
    name = "naive"

    def fit(self, y: pd.Series) -> "NaiveForecaster":
        self.last_date_ = y.index[-1]
        self.last_value_ = float(y.iloc[-1])
        diffs = y.diff().dropna()
        self.sigma_ = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
        return self

    def predict(self, h: int) -> pd.DataFrame:
        steps = np.arange(1, h + 1)
        mean = np.full(h, self.last_value_)
        sigma = self.sigma_ * np.sqrt(steps)
        df = normal_intervals(mean, sigma)
        df.index = future_index(self.last_date_, h)
        return df


class SeasonalNaiveForecaster:
    name = "seasonal_naive"

    def __init__(self, m: int = 12):
        self.m = m

    def fit(self, y: pd.Series) -> "SeasonalNaiveForecaster":
        self.last_date_ = y.index[-1]
        values = y.to_numpy()
        if len(values) > self.m:
            resid = values[self.m:] - values[:-self.m]
        else:
            resid = np.diff(values)
        self.sigma_ = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
        self.season_values_ = values[-self.m:]
        return self

    def predict(self, h: int) -> pd.DataFrame:
        reps = int(np.ceil(h / self.m))
        mean = np.tile(self.season_values_, reps)[:h]
        steps = np.arange(1, h + 1)
        sigma = self.sigma_ * np.sqrt((steps - 1) // self.m + 1)
        df = normal_intervals(mean, sigma)
        df.index = future_index(self.last_date_, h)
        return df


class HistoricalMeanForecaster:
    name = "historical_mean"

    def __init__(self, window: int | None = None):
        self.window = window

    def fit(self, y: pd.Series) -> "HistoricalMeanForecaster":
        series = y if self.window is None else y.iloc[-self.window:]
        self.last_date_ = y.index[-1]
        self.mean_ = float(series.mean())
        self.sigma_ = float(series.std(ddof=1)) if len(series) > 1 else 0.0
        self.T_ = len(series)
        return self

    def predict(self, h: int) -> pd.DataFrame:
        mean = np.full(h, self.mean_)
        sigma = np.full(h, self.sigma_ * np.sqrt(1 + 1 / self.T_))
        df = normal_intervals(mean, sigma)
        df.index = future_index(self.last_date_, h)
        return df


class DriftForecaster:
    name = "drift"

    def fit(self, y: pd.Series) -> "DriftForecaster":
        self.last_date_ = y.index[-1]
        self.T_ = len(y)
        self.last_value_ = float(y.iloc[-1])
        first_value = float(y.iloc[0])
        self.slope_ = (self.last_value_ - first_value) / (self.T_ - 1) if self.T_ > 1 else 0.0
        diffs = y.diff().dropna().to_numpy()
        resid = diffs - self.slope_
        self.sigma_ = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
        return self

    def predict(self, h: int) -> pd.DataFrame:
        steps = np.arange(1, h + 1)
        mean = self.last_value_ + steps * self.slope_
        sigma = self.sigma_ * np.sqrt(steps * (1 + steps / self.T_))
        df = normal_intervals(mean, sigma)
        df.index = future_index(self.last_date_, h)
        return df


MODELS: dict[str, type] = {
    "naive": NaiveForecaster,
    "seasonal_naive": SeasonalNaiveForecaster,
    "historical_mean": HistoricalMeanForecaster,
    "drift": DriftForecaster,
}


def build_model(name: str, **params) -> "NaiveForecaster | SeasonalNaiveForecaster | HistoricalMeanForecaster | DriftForecaster":
    return MODELS[name](**params)
