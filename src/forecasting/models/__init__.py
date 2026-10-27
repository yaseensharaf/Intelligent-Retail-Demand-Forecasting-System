from forecasting.models.base import INTERVAL_COLUMNS, Forecaster, future_index, normal_intervals
from forecasting.models.baselines import (
    MODELS,
    DriftForecaster,
    HistoricalMeanForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)
from forecasting.models.ets import ETSForecaster
from forecasting.models.hybrid import SarimaLstmHybridForecaster
from forecasting.models.intermittent import CrostonForecaster, TSBForecaster
from forecasting.models.sarima import SARIMAForecaster

# `MODELS` is deliberately mutated in place (not rebuilt as a new dict) so
# that it stays the exact same object as `forecasting.models.baselines.MODELS`
# -- tests monkeypatch that object directly (see test_backtest.py's
# `spy_registered` fixture) and expect the patch to be visible here too.
MODELS.update({
    "ets": ETSForecaster,
    "sarima": SARIMAForecaster,
    "sarima_lstm_hybrid": SarimaLstmHybridForecaster,
    "croston": CrostonForecaster,
    "tsb": TSBForecaster,
})


def build_model(name: str, **params):
    return MODELS[name](**params)


__all__ = [
    "INTERVAL_COLUMNS",
    "Forecaster",
    "future_index",
    "normal_intervals",
    "MODELS",
    "DriftForecaster",
    "HistoricalMeanForecaster",
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "ETSForecaster",
    "SARIMAForecaster",
    "SarimaLstmHybridForecaster",
    "CrostonForecaster",
    "TSBForecaster",
    "build_model",
]
