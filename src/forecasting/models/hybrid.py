"""The corrected SARIMA+LSTM hybrid (AGENTS.md "Correct hybrid"):

1. Fit SARIMA on train. Residuals = in-sample `fit.resid`, with the first
   `season_length + d` burn-in values dropped.
2. Train an LSTM on sliding windows of those residuals, with its own
   scaler fit only on train residuals.
3. Forecast residuals recursively h steps ahead from the last `window`
   train residuals.
4. Final = SARIMA forecast + blend_weight * residual forecast. The blend
   weight is grid-searched on a held-out validation tail carved from the
   end of train, not hand-picked -- `split_at` already guarantees `train`
   never reaches past the backtest origin, so carving a tail *inside*
   train cannot leak. Both the validation-fold models and the final
   full-train models are fit separately: scoring the blend weight against
   the same residuals the weight is chosen from would be optimistically
   biased, especially for an LSTM that can fit noise.

TensorFlow is imported lazily (inside functions, not at module scope) so
merely importing `forecasting.models` doesn't pay TF's cold-import cost.
RNGs are reseeded at the top of `fit()`, not once globally, because the
leakage test fits the same model twice in one process and a shared RNG
stream would make the second call diverge from the first for reasons that
have nothing to do with real leakage.
"""
from __future__ import annotations

import logging
import random

import numpy as np
import pandas as pd

from forecasting.models.base import Z95, future_index, normal_intervals
from forecasting.models.sarima import SARIMAForecaster

logger = logging.getLogger(__name__)

SEASON_LENGTH = 12
DEFAULT_BLEND_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)

_tf_module = None


def _lazy_tf():
    global _tf_module
    if _tf_module is None:
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
        _tf_module = tf
    return _tf_module


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    _lazy_tf().keras.utils.set_random_seed(seed)


def _sarima_in_sample_residuals(sarima: SARIMAForecaster, season_length: int) -> np.ndarray:
    resid = sarima.in_sample_residuals()
    burn_in = season_length + sarima.difference_order
    return resid[burn_in:] if len(resid) > burn_in else resid


def _build_lstm(window: int, units: int):
    _lazy_tf()  # ensures TF logging/determinism setup runs before keras is imported
    from tensorflow import keras

    model = keras.Sequential([
        keras.layers.Input(shape=(window, 1)),
        keras.layers.LSTM(units),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def _fit_lstm_residual_model(residuals: np.ndarray, window: int, units: int, epochs: int, patience: int):
    if len(residuals) <= window + 1:
        return None

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaled = scaler.fit_transform(residuals.reshape(-1, 1)).flatten()

    X, Y = [], []
    for i in range(len(scaled) - window):
        X.append(scaled[i:i + window])
        Y.append(scaled[i + window])
    X = np.asarray(X).reshape(-1, window, 1)
    Y = np.asarray(Y)

    model = _build_lstm(window, units)
    fit_kwargs: dict = {"epochs": epochs, "verbose": 0}
    if len(X) >= 10:
        from tensorflow import keras

        fit_kwargs["validation_split"] = 0.2
        fit_kwargs["callbacks"] = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ]
    model.fit(X, Y, **fit_kwargs)
    return model, scaler


def _forecast_residuals_recursive(lstm_state, residuals: np.ndarray, h: int, window: int) -> np.ndarray:
    if lstm_state is None:
        return np.zeros(h)
    model, scaler = lstm_state

    tail = residuals[-window:] if len(residuals) >= window else residuals
    buffer = list(scaler.transform(tail.reshape(-1, 1)).flatten())

    preds_scaled = []
    for _ in range(h):
        x = np.asarray(buffer[-window:]).reshape(1, window, 1)
        pred = float(model.predict(x, verbose=0)[0, 0])
        preds_scaled.append(pred)
        buffer.append(pred)

    preds_scaled = np.asarray(preds_scaled).reshape(-1, 1)
    return scaler.inverse_transform(preds_scaled).flatten()


def _select_blend_weight(actual: np.ndarray, sarima_fc: np.ndarray, resid_fc: np.ndarray, grid) -> float:
    best_weight, best_error = grid[0], np.inf
    for weight in grid:
        pred = sarima_fc + weight * resid_fc
        error = float(np.mean(np.abs(actual - pred)))
        if error < best_error:
            best_error, best_weight = error, weight
    return best_weight


class SarimaLstmHybridForecaster:
    name = "sarima_lstm_hybrid"

    def __init__(
        self,
        m: int = SEASON_LENGTH,
        window: int = 12,
        units: int = 8,
        epochs: int = 5,
        patience: int = 1,
        validation_h: int = 12,
        blend_grid=DEFAULT_BLEND_GRID,
        seed: int = 42,
    ):
        self.m = m
        self.window = window
        self.units = units
        self.epochs = epochs
        self.patience = patience
        self.validation_h = validation_h
        self.blend_grid = blend_grid
        self.seed = seed

    def fit(self, y: pd.Series) -> "SarimaLstmHybridForecaster":
        _seed_all(self.seed)
        self.last_date_ = y.index[-1]

        min_len_for_validation = self.window + self.validation_h + 2 * self.m
        if len(y) >= min_len_for_validation:
            train_inner = y.iloc[:-self.validation_h]
            valid_actual = y.iloc[-self.validation_h:].to_numpy()

            sarima_inner = SARIMAForecaster(m=self.m).fit(train_inner)
            resid_inner = _sarima_in_sample_residuals(sarima_inner, self.m)
            lstm_inner = _fit_lstm_residual_model(resid_inner, self.window, self.units, self.epochs, self.patience)

            sarima_fc_inner = sarima_inner.predict(self.validation_h)["mean"].to_numpy()
            resid_fc_inner = _forecast_residuals_recursive(lstm_inner, resid_inner, self.validation_h, self.window)
            self.blend_weight_ = _select_blend_weight(valid_actual, sarima_fc_inner, resid_fc_inner, self.blend_grid)
        else:
            self.blend_weight_ = 0.5

        # Final models are refit on the FULL train -- scoring the blend
        # weight against the same residuals it was chosen from would be
        # optimistically biased.
        self.sarima_ = SARIMAForecaster(m=self.m).fit(y)
        self.resid_ = _sarima_in_sample_residuals(self.sarima_, self.m)
        self.lstm_ = _fit_lstm_residual_model(self.resid_, self.window, self.units, self.epochs, self.patience)
        self.resid_sigma_ = float(np.std(self.resid_, ddof=1)) if len(self.resid_) > 1 else 0.0
        return self

    def predict(self, h: int) -> pd.DataFrame:
        sarima_forecast = self.sarima_.predict(h)
        resid_forecast = _forecast_residuals_recursive(self.lstm_, self.resid_, h, self.window)

        mean = sarima_forecast["mean"].to_numpy() + self.blend_weight_ * resid_forecast
        sarima_sigma = (sarima_forecast["hi95"] - sarima_forecast["lo95"]).to_numpy() / (2 * Z95)
        sigma = np.sqrt(sarima_sigma ** 2 + (self.blend_weight_ * self.resid_sigma_) ** 2)

        df = normal_intervals(mean, sigma)
        df.index = future_index(self.last_date_, h)
        return df
