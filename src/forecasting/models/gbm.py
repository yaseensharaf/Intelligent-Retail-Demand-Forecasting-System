"""Global LightGBM model, trained once across all products of a store.

This is deliberately *not* a `Forecaster`: `backtest_series`/`backtest_store`
only ever fit one product's series at a time, and forcing a panel-trained
model through that single-series seam would mean building an adapter whose
only job is to fake it. Instead `backtest.backtest_global` is the seam --
see that module. `fit_global_lightgbm` trains one booster on the whole
panel (features.build_panel, bounded by `train_end`); `FittedGlobalModel`
is the plain result object `predict_product` recurses from.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from forecasting import features
from forecasting.models.base import INTERVAL_COLUMNS, Z80, Z95, future_index

logger = logging.getLogger(__name__)

DEFAULT_VALIDATION_H = 12
DEFAULT_NUM_BOOST_ROUND = 200
DEFAULT_MIN_DATA_IN_LEAF = 10
DEFAULT_SEED = 42


@dataclass
class FittedGlobalModel:
    booster: lgb.Booster
    categories: list[str]
    feature_cols: list[str]
    lags: tuple[int, ...]
    windows: tuple[int, ...]
    sigma_by_step: np.ndarray = field(repr=False)

    def predict_product(self, product: str, history: pd.Series, h: int) -> pd.DataFrame:
        if product not in self.categories:
            raise ValueError(f"product {product!r} was not part of the training panel")
        code = self.categories.index(product)

        buffer = history.copy()
        preds: list[float] = []
        for _ in range(h):
            next_date = buffer.index[-1] + pd.offsets.MonthEnd(1)
            row = features.next_step_row(buffer, next_date, code, self.lags, self.windows)
            row = row[self.feature_cols]
            pred = max(float(self.booster.predict(row)[0]), 0.0)
            preds.append(pred)
            buffer = pd.concat([buffer, pd.Series([pred], index=[next_date])])

        mean = np.asarray(preds, dtype=float)
        steps = np.arange(1, h + 1)
        sigma = self.sigma_by_step[np.minimum(steps - 1, len(self.sigma_by_step) - 1)]
        df = pd.DataFrame({
            "mean": mean,
            "lo80": np.clip(mean - Z80 * sigma, 0.0, None),
            "hi80": mean + Z80 * sigma,
            "lo95": np.clip(mean - Z95 * sigma, 0.0, None),
            "hi95": mean + Z95 * sigma,
        })[INTERVAL_COLUMNS]
        df.index = future_index(history.index[-1], h)
        return df


def _per_step_conformal_sigma(
    booster: lgb.Booster,
    hist: pd.DataFrame,
    categories: list[str],
    feature_cols: list[str],
    lags: tuple[int, ...],
    windows: tuple[int, ...],
    validation_h: int,
) -> np.ndarray:
    """Residuals from running the *same* recursive mechanism over a
    held-out validation tail, per step -- a flat 1-step residual quantile
    under-covers once a step's errors compound through earlier recursive
    predictions, so the interval must widen with step from real evidence.
    """
    max_lag = max(lags) if lags else 0
    if len(hist.index) <= validation_h + max_lag + 1:
        flat = float(np.nanstd(hist.to_numpy())) or 1.0
        return np.full(validation_h, flat)

    history_cutoff = hist.index[-validation_h - 1]
    partial = FittedGlobalModel(booster, categories, feature_cols, lags, windows, np.zeros(validation_h))

    errors_by_step: list[list[float]] = [[] for _ in range(validation_h)]
    for product in categories:
        series = hist[product]
        history = series.loc[series.index <= history_cutoff]
        actual = series.loc[series.index > history_cutoff].iloc[:validation_h]
        if len(actual) < validation_h or len(history) == 0:
            continue
        forecast = partial.predict_product(product, history, validation_h)
        errors = actual.to_numpy() - forecast["mean"].to_numpy()
        for step, err in enumerate(errors):
            errors_by_step[step].append(err)

    sigma = np.array([
        float(np.std(errs, ddof=1)) if len(errs) > 1 else (float(abs(errs[0])) if errs else np.nan)
        for errs in errors_by_step
    ])
    for i in range(len(sigma)):
        if not np.isfinite(sigma[i]) or sigma[i] == 0:
            sigma[i] = sigma[i - 1] if i > 0 else 1.0
    return sigma


def fit_global_lightgbm(
    monthly: pd.DataFrame,
    train_end: pd.Timestamp,
    params: dict[str, Any] | None = None,
) -> FittedGlobalModel:
    params = params or {}
    lags = tuple(params.get("lags", features.DEFAULT_LAGS))
    windows = tuple(params.get("windows", features.DEFAULT_WINDOWS))
    seed = int(params.get("seed", DEFAULT_SEED))
    num_boost_round = int(params.get("num_boost_round", DEFAULT_NUM_BOOST_ROUND))
    validation_h = int(params.get("validation_h", DEFAULT_VALIDATION_H))
    min_data_in_leaf = int(params.get("min_data_in_leaf", DEFAULT_MIN_DATA_IN_LEAF))

    hist = monthly.loc[monthly.index <= train_end]
    panel, categories = features.build_panel(hist, train_end, lags=lags, windows=windows)
    feature_cols = features.feature_names(lags, windows)

    if panel.empty:
        raise ValueError("not enough history to build any training rows for the global model")

    cutoff = hist.index[-1] - pd.DateOffset(months=validation_h) if len(hist.index) > validation_h else None
    fit_panel = panel[panel["date"] <= cutoff] if cutoff is not None else panel
    if fit_panel.empty:
        fit_panel = panel

    train_set = lgb.Dataset(
        fit_panel[feature_cols],
        label=fit_panel["target"],
        categorical_feature=["product_code"],
        free_raw_data=False,
    )
    booster = lgb.train(
        {
            "objective": "regression",
            "metric": "l1",
            "verbosity": -1,
            "seed": seed,
            "deterministic": True,
            "num_threads": 1,
            "min_data_in_leaf": max(1, min_data_in_leaf),
        },
        train_set,
        num_boost_round=num_boost_round,
    )

    sigma_by_step = _per_step_conformal_sigma(
        booster, hist, categories, feature_cols, lags, windows, validation_h,
    )

    return FittedGlobalModel(
        booster=booster,
        categories=categories,
        feature_cols=feature_cols,
        lags=lags,
        windows=windows,
        sigma_by_step=sigma_by_step,
    )
