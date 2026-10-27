from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.models.base import INTERVAL_COLUMNS
from forecasting.models.hybrid import (
    SarimaLstmHybridForecaster,
    _sarima_in_sample_residuals,
    _select_blend_weight,
)
from forecasting.models.sarima import SARIMAForecaster


def _series(n: int, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2016-01-31", periods=n, freq="ME")
    trend = np.linspace(20, 30, n)
    season = 4 * np.sin(2 * np.pi * np.arange(n) / 12)
    noise = rng.normal(scale=1.0, size=n)
    return pd.Series(trend + season + noise, index=index)


class _StubSarima:
    def __init__(self, resid: np.ndarray, difference_order: int):
        self._resid = resid
        self.difference_order = difference_order

    def in_sample_residuals(self) -> np.ndarray:
        return self._resid


def test_sarima_in_sample_residuals_drops_season_length_plus_d_burn_in():
    resid = np.arange(30, dtype=float)
    stub = _StubSarima(resid, difference_order=1)

    trimmed = _sarima_in_sample_residuals(stub, season_length=12)

    np.testing.assert_allclose(trimmed, resid[13:])  # burn-in = 12 + 1


def test_sarima_in_sample_residuals_falls_back_to_untrimmed_when_too_short():
    resid = np.arange(5, dtype=float)
    stub = _StubSarima(resid, difference_order=1)

    trimmed = _sarima_in_sample_residuals(stub, season_length=12)

    np.testing.assert_allclose(trimmed, resid)


def test_select_blend_weight_picks_zero_when_residual_forecast_is_noise():
    actual = np.array([10.0, 11.0, 12.0])
    sarima_fc = np.array([10.0, 11.0, 12.0])  # perfect
    resid_fc = np.array([100.0, -100.0, 100.0])  # pure noise, hurts if weighted up

    weight = _select_blend_weight(actual, sarima_fc, resid_fc, grid=(0.0, 0.5, 1.0))
    assert weight == 0.0


def test_select_blend_weight_picks_one_when_residual_forecast_fixes_a_bias():
    actual = np.array([15.0, 16.0, 17.0])
    sarima_fc = np.array([10.0, 11.0, 12.0])  # consistently 5 low
    resid_fc = np.array([5.0, 5.0, 5.0])  # exactly corrects the bias

    weight = _select_blend_weight(actual, sarima_fc, resid_fc, grid=(0.0, 0.5, 1.0))
    assert weight == 1.0


def test_hybrid_fit_predict_shape_columns_date_alignment_and_nested_intervals():
    # one fit shared across both checks -- each SARIMA+LSTM fit is
    # expensive, and these two properties don't need separate models.
    y = _series(48)
    model = SarimaLstmHybridForecaster(epochs=2).fit(y)
    forecast = model.predict(6)

    assert len(forecast) == 6
    assert list(forecast.columns) == INTERVAL_COLUMNS
    assert forecast.index[0] == y.index[-1] + pd.offsets.MonthEnd(1)

    assert (forecast["lo95"] <= forecast["lo80"]).all()
    assert (forecast["lo80"] <= forecast["mean"]).all()
    assert (forecast["mean"] <= forecast["hi80"]).all()
    assert (forecast["hi80"] <= forecast["hi95"]).all()


def test_hybrid_blend_weight_is_selected_from_a_held_out_validation_tail():
    y = _series(60)  # long enough to clear min_len_for_validation (12+12+24=48)
    model = SarimaLstmHybridForecaster(epochs=2).fit(y)

    assert model.blend_weight_ in model.blend_grid


def test_hybrid_degrades_gracefully_on_short_series():
    y = _series(20)  # too short for the internal validation tail
    model = SarimaLstmHybridForecaster(epochs=2).fit(y)

    assert model.blend_weight_ == 0.5
    assert model.lstm_ is None  # too short for any sliding window either


def test_hybrid_two_fits_in_one_process_are_deterministic():
    y = _series(40)

    forecast_1 = SarimaLstmHybridForecaster(epochs=2).fit(y).predict(6)
    forecast_2 = SarimaLstmHybridForecaster(epochs=2).fit(y).predict(6)

    pd.testing.assert_frame_equal(forecast_1, forecast_2, check_exact=False, rtol=1e-4)


def test_hybrid_poisoned_future_does_not_change_the_forecast():
    y = _series(48)
    origin_ts = y.index[35]
    train = y.loc[:origin_ts]

    poisoned = y.copy()
    poisoned.loc[poisoned.index > origin_ts] = 1e6
    train_poisoned = poisoned.loc[:origin_ts]

    pd.testing.assert_series_equal(train, train_poisoned)

    forecast_clean = SarimaLstmHybridForecaster(epochs=2).fit(train).predict(6)
    forecast_poisoned = SarimaLstmHybridForecaster(epochs=2).fit(train_poisoned).predict(6)
    pd.testing.assert_frame_equal(forecast_clean, forecast_poisoned, check_exact=False, rtol=1e-4)


def test_hybrid_sigma_includes_both_sarima_and_residual_uncertainty():
    y = _series(40)  # below the validation-fold threshold -- this test only checks sigma composition
    hybrid = SarimaLstmHybridForecaster(epochs=2).fit(y)
    sarima_alone = SARIMAForecaster().fit(y)

    hybrid_forecast = hybrid.predict(6)
    sarima_forecast = sarima_alone.predict(6)

    if hybrid.resid_sigma_ > 0 and hybrid.blend_weight_ > 0:
        hybrid_width = (hybrid_forecast["hi95"] - hybrid_forecast["lo95"]).to_numpy()
        sarima_width = (sarima_forecast["hi95"] - sarima_forecast["lo95"]).to_numpy()
        assert (hybrid_width >= sarima_width - 1e-6).all()
