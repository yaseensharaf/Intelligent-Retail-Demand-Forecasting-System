from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecasting import cli


def test_backtest_and_report_create_outputs_and_name_all_baselines(tiny_config_path, tiny_config):
    assert cli.main(["backtest", "--store", "tiny", "--config", str(tiny_config_path)]) == 0

    outputs = Path(tiny_config["paths"]["outputs"])
    assert (outputs / "backtest_tiny_forecasts.csv").exists()
    assert (outputs / "backtest_tiny_scores.csv").exists()
    assert (outputs / "data_quality.json").exists()

    assert cli.main(["report", "--store", "tiny", "--config", str(tiny_config_path)]) == 0

    report_path = Path(tiny_config["paths"]["report"])
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    for model in ["naive", "seasonal_naive", "historical_mean", "drift"]:
        assert model in content


def test_run_requires_a_prior_backtest(tiny_config_path):
    with pytest.raises(FileNotFoundError, match="run `backtest` first"):
        cli.main(["run", "--store", "tiny", "--config", str(tiny_config_path)])


def test_run_writes_forecast_with_correctly_aligned_dates(tiny_config_path, tiny_config):
    assert cli.main(["backtest", "--store", "tiny", "--config", str(tiny_config_path)]) == 0
    assert cli.main(["run", "--store", "tiny", "--config", str(tiny_config_path)]) == 0

    outputs = Path(tiny_config["paths"]["outputs"])
    forecast_path = outputs / "forecast_tiny.csv"
    assert forecast_path.exists()

    forecast = pd.read_csv(forecast_path, parse_dates=["date"])
    n_products = 3
    expected_h = 12  # tiny_config: data.end 2023-12 -> forecast.end 2024-12
    assert len(forecast) == n_products * expected_h
    assert forecast["date"].min() == pd.Timestamp("2024-01-31")
    assert forecast["date"].max() == pd.Timestamp("2024-12-31")
    assert set(forecast["model_used"]).issubset(set(tiny_config["models"]))


def test_run_products_flag_limits_forecast_to_first_n_products(tiny_config_path, tiny_config):
    assert cli.main(["backtest", "--store", "tiny", "--config", str(tiny_config_path)]) == 0
    assert cli.main(["run", "--store", "tiny", "--config", str(tiny_config_path), "--products", "1"]) == 0

    outputs = Path(tiny_config["paths"]["outputs"])
    forecast = pd.read_csv(outputs / "forecast_tiny.csv")
    assert forecast["product"].nunique() == 1


def test_unknown_store_raises(tiny_config_path):
    with pytest.raises(ValueError, match="unknown store"):
        cli.main(["backtest", "--store", "nope", "--config", str(tiny_config_path)])


@pytest.mark.skipif(not Path("data/raw/hm.csv").exists(), reason="real data not present")
def test_hm_historical_mean_matches_baseline_check():
    from forecasting import data as fdata
    from forecasting.config import load_config

    cfg = load_config()
    store = fdata.load_store("hm", cfg)
    train, test = store.monthly.loc[:"2022"], store.monthly.loc["2023"]

    errors = []
    for product in test.columns:
        actual = test[product].to_numpy()
        forecast = np.full(len(actual), train[product].mean())
        errors.append(np.mean(np.abs(actual - forecast) / actual) * 100)

    assert np.mean(errors) == pytest.approx(12.03, abs=0.5)
