from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from forecasting import backtest
from forecasting.data import StoreData
from forecasting.models.base import INTERVAL_COLUMNS
from forecasting.models.gbm import fit_global_lightgbm

FAST_PARAMS = {
    "lags": [1, 2, 3, 12],
    "windows": [3, 6],
    "num_boost_round": 20,
    "validation_h": 6,
    "min_data_in_leaf": 2,
}


def _synthetic_monthly(n_products: int = 4, n_months: int = 48, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2018-01-31", periods=n_months, freq="ME")
    data = {}
    for p in range(n_products):
        trend = np.linspace(10, 10 + p, n_months)
        season = 3 * np.sin(2 * np.pi * np.arange(n_months) / 12)
        noise = rng.normal(scale=0.5, size=n_months)
        data[f"product_{p}"] = np.clip(trend + season + noise, 0, None)
    return pd.DataFrame(data, index=index)


def _store(monthly: pd.DataFrame) -> StoreData:
    return StoreData(key="t", name="t", monthly=monthly, categories={}, quality=None)


def test_fit_global_lightgbm_predict_product_shape_and_columns():
    monthly = _synthetic_monthly()
    fitted = fit_global_lightgbm(monthly, monthly.index[35], FAST_PARAMS)

    forecast = fitted.predict_product("product_0", monthly["product_0"].loc[:monthly.index[35]], h=6)

    assert len(forecast) == 6
    assert list(forecast.columns) == INTERVAL_COLUMNS
    assert forecast.index[0] == monthly.index[35] + pd.offsets.MonthEnd(1)


def test_predict_product_intervals_are_nested_and_non_negative():
    monthly = _synthetic_monthly()
    fitted = fit_global_lightgbm(monthly, monthly.index[35], FAST_PARAMS)
    forecast = fitted.predict_product("product_1", monthly["product_1"].loc[:monthly.index[35]], h=6)

    assert (forecast["lo95"] <= forecast["lo80"]).all()
    assert (forecast["lo80"] <= forecast["mean"]).all()
    assert (forecast["mean"] <= forecast["hi80"]).all()
    assert (forecast["hi80"] <= forecast["hi95"]).all()
    assert (forecast["lo80"] >= 0).all()
    assert (forecast["lo95"] >= 0).all()


def test_predict_product_rejects_unseen_product():
    monthly = _synthetic_monthly()
    fitted = fit_global_lightgbm(monthly, monthly.index[35], FAST_PARAMS)

    with pytest.raises(ValueError, match="not part of the training panel"):
        fitted.predict_product("nonexistent", monthly["product_0"], h=3)


def test_categories_are_fixed_sorted_product_list():
    monthly = _synthetic_monthly(n_products=3)
    fitted = fit_global_lightgbm(monthly, monthly.index[35], FAST_PARAMS)

    assert fitted.categories == sorted(monthly.columns)


def test_sigma_by_step_widens_or_holds_relative_to_first_step():
    monthly = _synthetic_monthly()
    fitted = fit_global_lightgbm(monthly, monthly.index[35], FAST_PARAMS)

    # a flat 1-step residual quantile would under-cover once errors compound
    # recursively -- the later steps must not be narrower than the first.
    assert fitted.sigma_by_step[-1] >= fitted.sigma_by_step[0] * 0.5


def test_backtest_global_produces_expected_columns_and_row_count():
    monthly = _synthetic_monthly(n_products=3, n_months=48)
    store = _store(monthly)
    origin = str(monthly.index[35])[:7]

    forecasts = backtest.backtest_global(store, "lightgbm_global", [origin], h=6, params=FAST_PARAMS)

    assert set(forecasts.columns) == {
        "product", "model", "origin", "date", "step", "actual",
        "mean", "lo80", "hi80", "lo95", "hi95",
    }
    assert len(forecasts) == 3 * 6
    assert (forecasts["model"] == "lightgbm_global").all()


def test_backtest_global_poisoned_future_does_not_change_any_products_forecast():
    monthly = _synthetic_monthly(n_products=4, n_months=48)
    store = _store(monthly)
    origin = str(monthly.index[35])[:7]
    origin_ts = backtest._to_month_end(origin)

    poisoned_monthly = monthly.copy()
    poisoned_monthly.loc[poisoned_monthly.index > origin_ts] = 1e6
    poisoned_store = replace(store, monthly=poisoned_monthly)

    clean = backtest.backtest_global(store, "lightgbm_global", [origin], h=6, params=FAST_PARAMS)
    poisoned = backtest.backtest_global(poisoned_store, "lightgbm_global", [origin], h=6, params=FAST_PARAMS)

    compare_cols = ["product", "model", "origin", "date", "step", *INTERVAL_COLUMNS]
    pd.testing.assert_frame_equal(
        clean[compare_cols].reset_index(drop=True),
        poisoned[compare_cols].reset_index(drop=True),
    )
