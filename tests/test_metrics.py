from __future__ import annotations

import numpy as np
import pytest

from forecasting import metrics


def test_mae_hand_computed():
    assert metrics.mae([10, 20, 30], [12, 18, 33]) == pytest.approx((2 + 2 + 3) / 3)


def test_rmse_hand_computed():
    assert metrics.rmse([10, 20], [12, 16]) == pytest.approx(np.sqrt((4 + 16) / 2))


def test_wape_hand_computed():
    # errors 2, 4; actuals 10, 20 -> wape = 6/30*100
    assert metrics.wape([10, 20], [12, 16]) == pytest.approx(20.0)


def test_pooled_wape_sums_across_groups_rather_than_averaging_ratios():
    # product A: actual 100, error 10 -> ratio 10%; product B: actual 1, error 1 -> ratio 100%.
    # A mean-of-ratios WAPE would be dominated by B's 100%; pooled WAPE is
    # (10+1)/(100+1) ~= 10.9%, which is what the headline report metric needs.
    y = [100, 1]
    f = [90, 2]
    assert metrics.pooled_wape(y, f) == pytest.approx(11 / 101 * 100)


def test_bias_positive_means_over_forecast():
    assert metrics.bias([10, 10], [12, 14]) == pytest.approx(3.0)
    assert metrics.bias([10, 10], [8, 6]) == pytest.approx(-3.0)


def test_smape_hand_computed():
    # |10-12|/(10+12) + |20-16|/(20+16) = 2/22 + 4/36; mean * 200
    expected = np.mean([2 / 22, 4 / 36]) * 200
    assert metrics.smape([10, 20], [12, 16]) == pytest.approx(expected)


def test_mape_hand_computed():
    assert metrics.mape([10, 20], [12, 16]) == pytest.approx(np.mean([2 / 10, 4 / 20]) * 100)


def test_mape_nan_when_actual_is_zero():
    assert np.isnan(metrics.mape([0, 20], [1, 16]))


def test_mase_equals_one_by_construction():
    # training seasonal (m=4) differences are a constant 5 -> scale = 5
    y_train = [10, 20, 30, 40, 15, 25, 35, 45]
    # test errors also average 5 -> MASE = 5 / 5 = 1
    y = [10, 20, 30]
    f = [15, 25, 35]
    assert metrics.mase(y, f, y_train, m=4) == pytest.approx(1.0)


def test_mase_nan_when_scale_is_zero():
    y_train = [10, 10, 10, 10, 10, 10, 10, 10]  # perfectly periodic -> scale 0
    assert np.isnan(metrics.mase([1], [2], y_train, m=4))


def test_interval_coverage():
    y = [1, 2, 3, 4]
    lo = [0, 0, 0, 10]
    hi = [5, 5, 5, 20]
    assert metrics.interval_coverage(y, lo, hi) == pytest.approx(0.75)


def test_diebold_mariano_identical_errors_gives_p_near_one():
    rng = np.random.default_rng(0)
    errors = rng.normal(size=100)
    stat, p_value = metrics.diebold_mariano(errors, errors, h=1)
    assert stat == pytest.approx(0.0)
    assert p_value == pytest.approx(1.0)


def test_diebold_mariano_detects_clearly_better_model():
    rng = np.random.default_rng(0)
    e1 = rng.normal(loc=0, scale=5, size=200)  # much larger errors
    e2 = rng.normal(loc=0, scale=0.5, size=200)  # much smaller errors
    _, p_value = metrics.diebold_mariano(e1, e2, h=1)
    assert p_value < 0.05


def test_all_metrics_returns_expected_keys():
    y_train = list(range(1, 25))
    result = metrics.all_metrics([10, 20], [12, 16], y_train, m=12)
    assert set(result) == {"mae", "rmse", "wape", "bias", "smape", "mape", "mase"}
    assert all(isinstance(v, float) for v in result.values())
