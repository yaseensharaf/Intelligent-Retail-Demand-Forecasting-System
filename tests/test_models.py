from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting.models import MODELS, INTERVAL_COLUMNS, build_model, future_index

TRAIN_INDEX = pd.date_range("2022-01-31", periods=12, freq="ME")


def _series(values) -> pd.Series:
    return pd.Series(values, index=TRAIN_INDEX, dtype=float)


@pytest.mark.parametrize("name", list(MODELS))
def test_predict_shape_and_columns(name):
    y = _series(np.arange(12) + 10.0)
    model = build_model(name).fit(y)
    forecast = model.predict(6)

    assert len(forecast) == 6
    assert list(forecast.columns) == INTERVAL_COLUMNS


@pytest.mark.parametrize("name", list(MODELS))
def test_first_forecast_date_is_month_after_training(name):
    y = _series(np.arange(12) + 10.0)
    model = build_model(name).fit(y)
    forecast = model.predict(3)

    assert forecast.index[0] == TRAIN_INDEX[-1] + pd.offsets.MonthEnd(1)


@pytest.mark.parametrize("name", list(MODELS))
def test_intervals_are_nested_around_mean(name):
    y = _series(np.arange(12) + 10.0 + np.sin(np.arange(12)))
    model = build_model(name).fit(y)
    forecast = model.predict(6)

    assert (forecast["lo95"] <= forecast["lo80"]).all()
    assert (forecast["lo80"] <= forecast["mean"]).all()
    assert (forecast["mean"] <= forecast["hi80"]).all()
    assert (forecast["hi80"] <= forecast["hi95"]).all()


def test_seasonal_naive_repeats_last_year():
    y = _series(np.arange(12) + 10.0)
    model = build_model("seasonal_naive", m=12).fit(y)
    forecast = model.predict(12)

    np.testing.assert_allclose(forecast["mean"].to_numpy(), y.to_numpy())


def test_historical_mean_is_flat():
    y = _series([10, 12, 11, 13, 10, 12, 11, 13, 10, 12, 11, 13])
    model = build_model("historical_mean").fit(y)
    forecast = model.predict(6)

    assert forecast["mean"].nunique() == 1
    assert forecast["mean"].iloc[0] == pytest.approx(y.mean())


def test_naive_forecasts_last_value():
    y = _series(np.arange(12) + 10.0)
    model = build_model("naive").fit(y)
    forecast = model.predict(4)

    np.testing.assert_allclose(forecast["mean"].to_numpy(), np.full(4, y.iloc[-1]))


def test_drift_extrapolates_the_line_from_first_to_last():
    y = _series(np.linspace(0, 11, 12))
    model = build_model("drift").fit(y)
    forecast = model.predict(2)

    slope = (y.iloc[-1] - y.iloc[0]) / (len(y) - 1)
    expected = y.iloc[-1] + slope * np.array([1, 2])
    np.testing.assert_allclose(forecast["mean"].to_numpy(), expected)


def test_future_index_starts_month_after_last_date():
    idx = future_index(pd.Timestamp("2023-12-31"), 3)
    assert list(idx) == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29"), pd.Timestamp("2024-03-31")]
