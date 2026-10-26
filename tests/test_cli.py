from __future__ import annotations

from pathlib import Path

import numpy as np
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


def test_run_prints_phase_2_notice(capsys):
    assert cli.main(["run"]) == 0
    assert "Phase 2" in capsys.readouterr().out


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
