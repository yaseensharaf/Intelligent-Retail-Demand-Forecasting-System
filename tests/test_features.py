from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting import features

INDEX = pd.date_range("2019-01-31", periods=36, freq="ME")


def _monthly(values_by_product: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(values_by_product, index=INDEX)


def test_build_panel_never_sees_rows_after_train_end():
    monthly = _monthly({"a": np.arange(36, dtype=float), "b": np.arange(36, dtype=float) * 2})
    train_end = INDEX[23]

    panel, categories = features.build_panel(monthly, train_end)

    assert categories == ["a", "b"]
    assert (panel["date"] <= train_end).all()


def test_build_panel_lag_1_matches_previous_month_value():
    monthly = _monthly({"a": np.arange(36, dtype=float)})
    panel, _ = features.build_panel(monthly, INDEX[-1])

    row = panel[panel["date"] == INDEX[20]].iloc[0]
    assert row["lag_1"] == pytest.approx(19.0)
    assert row["lag_2"] == pytest.approx(18.0)


def test_build_panel_rolling_mean_excludes_current_month():
    values = np.array([10.0, 20.0, 30.0, 40.0, 50.0] + [0.0] * 31)
    monthly = _monthly({"a": values})
    panel, _ = features.build_panel(monthly, INDEX[-1], lags=(1,), windows=(3,))

    row = panel[panel["date"] == INDEX[4]].iloc[0]  # value 50.0, preceded by 10,20,30,40
    assert row["roll_mean_3"] == pytest.approx(np.mean([20.0, 30.0, 40.0]))


def test_build_panel_product_code_is_stable_and_sorted():
    monthly = _monthly({"z-product": np.ones(36), "a-product": np.ones(36)})
    panel, categories = features.build_panel(monthly, INDEX[-1])

    assert categories == ["a-product", "z-product"]
    a_code = panel.loc[panel["product"] == "a-product", "product_code"].iloc[0]
    z_code = panel.loc[panel["product"] == "z-product", "product_code"].iloc[0]
    assert a_code == 0
    assert z_code == 1


def test_month_of_year_features_cycle():
    idx = pd.date_range("2023-01-31", periods=12, freq="ME")
    moy = features.month_of_year_features(idx)

    assert list(moy["month"]) == list(range(1, 13))
    np.testing.assert_allclose(moy["month_sin"].iloc[0], 0.5, atol=1e-6)


def test_next_step_row_uses_growing_buffer_for_lags_and_rolling():
    history = pd.Series(np.arange(1.0, 13.0), index=INDEX[:12])  # 1..12
    next_date = INDEX[12]

    row = features.next_step_row(history, next_date, product_code=0, lags=(1, 2), windows=(3,))

    assert row["lag_1"].iloc[0] == pytest.approx(12.0)
    assert row["lag_2"].iloc[0] == pytest.approx(11.0)
    assert row["roll_mean_3"].iloc[0] == pytest.approx(np.mean([10.0, 11.0, 12.0]))
    assert row["product_code"].iloc[0] == 0


def test_next_step_row_reflects_appended_recursive_prediction():
    history = pd.Series(np.arange(1.0, 13.0), index=INDEX[:12])
    extended = pd.concat([history, pd.Series([99.0], index=[INDEX[12]])])

    row = features.next_step_row(extended, INDEX[13], product_code=0, lags=(1,), windows=(3,))

    assert row["lag_1"].iloc[0] == pytest.approx(99.0)
    assert row["roll_mean_3"].iloc[0] == pytest.approx(np.mean([11.0, 12.0, 99.0]))


def test_feature_names_matches_build_panel_columns():
    monthly = _monthly({"a": np.arange(36, dtype=float)})
    panel, _ = features.build_panel(monthly, INDEX[-1], lags=(1, 2), windows=(3,))

    for name in features.feature_names(lags=(1, 2), windows=(3,)):
        assert name in panel.columns or name == "product_code"
