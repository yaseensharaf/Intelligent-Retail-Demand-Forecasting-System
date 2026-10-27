from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting.models.intermittent import CrostonForecaster, TSBForecaster

INDEX = pd.date_range("2015-01-31", periods=60, freq="ME")


def _periodic_series(period: int = 3, size: float = 30.0, n: int = 60) -> pd.Series:
    values = np.array([size if (i + 1) % period == 0 else 0.0 for i in range(n)])
    return pd.Series(values, index=INDEX[:n])


def _all_zero_series(n: int = 24) -> pd.Series:
    return pd.Series(np.zeros(n), index=INDEX[:n])


def test_croston_converges_to_demand_size_over_interval():
    y = _periodic_series(period=3, size=30.0)
    model = CrostonForecaster(alpha=0.1).fit(y)
    forecast = model.predict(6)

    # demand size 30 every 3 months -> steady-state level = 30 / 3 = 10
    assert forecast["mean"].iloc[0] == pytest.approx(10.0, abs=0.5)
    assert forecast["mean"].nunique() == 1  # flat forecast


def test_tsb_forecasts_a_positive_flat_level_for_periodic_demand():
    y = _periodic_series(period=3, size=30.0)
    model = TSBForecaster(alpha=0.1, beta=0.1).fit(y)
    forecast = model.predict(6)

    assert forecast["mean"].nunique() == 1
    assert 0.0 < forecast["mean"].iloc[0] < 30.0


def test_croston_handles_series_with_no_demand_at_all():
    y = _all_zero_series()
    model = CrostonForecaster().fit(y)
    forecast = model.predict(6)

    np.testing.assert_allclose(forecast["mean"].to_numpy(), np.zeros(6))


def test_tsb_handles_series_with_no_demand_at_all():
    y = _all_zero_series()
    model = TSBForecaster().fit(y)
    forecast = model.predict(6)

    np.testing.assert_allclose(forecast["mean"].to_numpy(), np.zeros(6))


@pytest.mark.parametrize("cls", [CrostonForecaster, TSBForecaster])
def test_intervals_are_clipped_at_zero(cls):
    y = _periodic_series(period=6, size=5.0)  # sparse, low-volume -> wide relative interval
    model = cls().fit(y)
    forecast = model.predict(6)

    assert (forecast["lo80"] >= 0.0).all()
    assert (forecast["lo95"] >= 0.0).all()


@pytest.mark.parametrize("cls", [CrostonForecaster, TSBForecaster])
def test_intervals_stay_nested_after_clipping(cls):
    y = _periodic_series(period=4, size=8.0)
    model = cls().fit(y)
    forecast = model.predict(6)

    assert (forecast["lo95"] <= forecast["lo80"]).all()
    assert (forecast["lo80"] <= forecast["mean"]).all()
    assert (forecast["mean"] <= forecast["hi80"]).all()
    assert (forecast["hi80"] <= forecast["hi95"]).all()


@pytest.mark.parametrize("cls", [CrostonForecaster, TSBForecaster])
def test_poisoned_future_does_not_change_forecast(cls):
    y = _periodic_series(period=3, size=30.0)
    origin = y.index[35]
    train = y.loc[:origin]

    poisoned = y.copy()
    poisoned.loc[poisoned.index > origin] = 1e6
    train_poisoned = poisoned.loc[:origin]

    pd.testing.assert_series_equal(train, train_poisoned)

    forecast_clean = cls().fit(train).predict(6)
    forecast_poisoned = cls().fit(train_poisoned).predict(6)
    pd.testing.assert_frame_equal(forecast_clean, forecast_poisoned)
