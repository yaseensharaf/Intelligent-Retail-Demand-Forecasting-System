from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting import backtest, data
from forecasting.models import MODELS, build_model
from forecasting.models.base import future_index, normal_intervals


def _series(n: int = 60, start: str = "2019-01-31") -> pd.Series:
    rng = np.random.default_rng(0)
    return pd.Series(10 + rng.normal(size=n), index=pd.date_range(start, periods=n, freq="ME"))


class SpyForecaster:
    name = "spy"
    fit_dates: list[pd.Timestamp] = []

    def fit(self, y: pd.Series) -> "SpyForecaster":
        self.last_date_ = y.index[-1]
        self.last_value_ = float(y.iloc[-1])
        SpyForecaster.fit_dates.append(self.last_date_)
        return self

    def predict(self, h: int) -> pd.DataFrame:
        mean = np.full(h, self.last_value_)
        df = normal_intervals(mean, np.zeros(h))
        df.index = future_index(self.last_date_, h)
        return df


@pytest.fixture
def spy_registered(monkeypatch):
    import forecasting.models.baselines as baselines_mod

    SpyForecaster.fit_dates = []
    monkeypatch.setitem(baselines_mod.MODELS, "spy", SpyForecaster)
    yield SpyForecaster


def test_split_at_rejects_leakage_into_training():
    y = _series()
    train, test = backtest.split_at(y, "2021-12", 12)

    assert train.index.max() < test.index.min()
    assert train.index.max() <= pd.Timestamp("2021-12-31")


def test_split_at_yields_exactly_h_consecutive_months():
    y = _series()
    for origin in ["2020-12", "2021-12", "2022-12"]:
        _, test = backtest.split_at(y, origin, 12)
        assert len(test) == 12
        expected = pd.date_range(test.index[0], periods=12, freq="ME")
        pd.testing.assert_index_equal(test.index, expected)


def test_spy_model_never_sees_data_after_origin(spy_registered):
    y = _series()
    origins = ["2020-12", "2021-12", "2022-12"]
    backtest.backtest_series("widget", y, ["spy"], origins, h=12)

    assert len(spy_registered.fit_dates) == len(origins)
    for origin, fit_date in zip(origins, spy_registered.fit_dates):
        origin_ts = backtest._to_month_end(origin)
        assert fit_date <= origin_ts


def test_poisoned_future_does_not_change_the_forecast():
    y = _series()
    origin = "2021-12"
    origin_ts = backtest._to_month_end(origin)

    train_clean, _ = backtest.split_at(y, origin, 12)

    poisoned = y.copy()
    poisoned.loc[poisoned.index > origin_ts] = 1e6
    train_poisoned, _ = backtest.split_at(poisoned, origin, 12)

    pd.testing.assert_series_equal(train_clean, train_poisoned)

    for name in MODELS:
        forecast_clean = build_model(name).fit(train_clean).predict(12)
        forecast_poisoned = build_model(name).fit(train_poisoned).predict(12)
        pd.testing.assert_frame_equal(forecast_clean, forecast_poisoned)


def test_backtest_series_produces_expected_columns():
    y = _series()
    forecasts = backtest.backtest_series("widget", y, ["naive", "historical_mean"], ["2021-12"], h=12)

    assert set(forecasts.columns) == {
        "product", "model", "origin", "date", "step", "actual",
        "mean", "lo80", "hi80", "lo95", "hi95",
    }
    assert len(forecasts) == 2 * 12


def test_backtest_store_and_score_and_summarise(tiny_config):
    store = data.load_store("tiny", tiny_config)
    forecasts = backtest.backtest_store(store, tiny_config)

    n_products = store.monthly.shape[1]
    n_origins = len(tiny_config["backtest"]["origins"])
    n_models = len(tiny_config["models"]) + len(tiny_config.get("global_models", []))
    h = tiny_config["backtest"]["horizon"]
    assert len(forecasts) == n_products * n_origins * n_models * h

    scores = backtest.score(forecasts, store.monthly, m=tiny_config["season_length"])
    assert len(scores) == n_products * n_origins * n_models
    for col in ["mae", "rmse", "wape", "bias", "smape", "mape", "mase", "coverage80", "coverage95"]:
        assert col in scores.columns

    summary = backtest.summarise(scores)
    assert set(summary["model"]) == set(tiny_config["models"]) | set(tiny_config.get("global_models", []))
    assert summary["wins_product_origin_pairs"].sum() == n_products * n_origins

    comparison = backtest.compare_to_reference(forecasts, reference="historical_mean")
    expected_models = (set(tiny_config["models"]) | set(tiny_config.get("global_models", []))) - {"historical_mean"}
    assert set(comparison["model"]) == expected_models
    assert (comparison["p_value"] >= 0).all() and (comparison["p_value"] <= 1).all()


def test_backtest_store_truncates_to_first_n_products(tiny_config):
    store = data.load_store("tiny", tiny_config)
    forecasts = backtest.backtest_store(store, tiny_config, products=1)

    n_origins = len(tiny_config["backtest"]["origins"])
    n_models = len(tiny_config["models"])
    h = tiny_config["backtest"]["horizon"]
    assert forecasts["product"].nunique() == 1
    assert len(forecasts) == 1 * n_origins * n_models * h


def test_pick_reference_baseline_returns_a_baseline_model(tiny_config):
    store = data.load_store("tiny", tiny_config)
    forecasts = backtest.backtest_store(store, tiny_config)
    scores = backtest.score(forecasts, store.monthly, m=tiny_config["season_length"])

    reference = backtest.pick_reference_baseline(scores)
    assert reference in backtest.BASELINE_MODELS
