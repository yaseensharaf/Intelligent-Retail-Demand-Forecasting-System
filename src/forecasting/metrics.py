"""Forecast accuracy metrics. Inputs are floats; never round before scoring."""
from __future__ import annotations

import numpy as np
from scipy import stats


def _as_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def mae(y, f) -> float:
    y, f = _as_array(y), _as_array(f)
    return float(np.mean(np.abs(y - f)))


def rmse(y, f) -> float:
    y, f = _as_array(y), _as_array(f)
    return float(np.sqrt(np.mean((y - f) ** 2)))


def wape(y, f) -> float:
    y, f = _as_array(y), _as_array(f)
    denom = np.sum(np.abs(y))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y - f)) / denom * 100)


def bias(y, f) -> float:
    y, f = _as_array(y), _as_array(f)
    return float(np.mean(f - y))


def smape(y, f) -> float:
    y, f = _as_array(y), _as_array(f)
    denom = np.abs(y) + np.abs(f)
    nonzero = denom != 0
    ratio = np.zeros_like(denom)
    ratio[nonzero] = np.abs(y[nonzero] - f[nonzero]) / denom[nonzero]
    return float(np.mean(ratio) * 200)


def mape(y, f) -> float:
    y, f = _as_array(y), _as_array(f)
    if np.any(y == 0):
        return float("nan")
    return float(np.mean(np.abs(y - f) / np.abs(y)) * 100)


def mase(y, f, y_train, m: int = 12) -> float:
    y, f, y_train = _as_array(y), _as_array(f), _as_array(y_train)
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    if scale == 0:
        return float("nan")
    return float(mae(y, f) / scale)


def interval_coverage(y, lo, hi) -> float:
    y, lo, hi = _as_array(y), _as_array(lo), _as_array(hi)
    return float(np.mean((y >= lo) & (y <= hi)))


def diebold_mariano(e1, e2, h: int = 1) -> tuple[float, float]:
    """Diebold-Mariano test on squared-error loss, with the Harvey-Leybourne-Newbold
    small-sample correction. Returns (stat, p_value); p_value near 1 means no
    detectable difference, small p_value means one model's errors are reliably smaller.
    """
    e1, e2 = _as_array(e1), _as_array(e2)
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    mean_d = float(np.mean(d))

    d_centered = d - mean_d
    gamma0 = float(np.sum(d_centered * d_centered) / n)
    if np.isclose(gamma0, 0.0):
        return 0.0, 1.0

    var_d = gamma0
    for k in range(1, h):
        gamma_k = float(np.sum(d_centered[:n - k] * d_centered[k:]) / n)
        var_d += 2 * gamma_k
    var_d /= n

    dm_stat = mean_d / np.sqrt(var_d)
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat *= correction

    p_value = float(2 * (1 - stats.t.cdf(np.abs(dm_stat), df=n - 1)))
    return float(dm_stat), p_value


def all_metrics(y, f, y_train, m: int = 12) -> dict[str, float]:
    return {
        "mae": mae(y, f),
        "rmse": rmse(y, f),
        "wape": wape(y, f),
        "bias": bias(y, f),
        "smape": smape(y, f),
        "mape": mape(y, f),
        "mase": mase(y, f, y_train, m=m),
    }
