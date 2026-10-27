"""Shared fixtures. Tests never touch real data under data/raw/."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

PRODUCTS = ["Lace-Up Boots", "Cotton Blazer", "Wool Trousers"]
START = "2019-01"
END = "2023-12"
EMPTY_MONTH_PRODUCT = PRODUCTS[0]
EMPTY_MONTH = "2021-06"
WRONG_SEASON_PRODUCT = PRODUCTS[1]
WRONG_SEASON_MONTH = "2020-03"


def _season_of(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


@pytest.fixture
def tiny_transactions_csv(tmp_path: Path) -> Path:
    months = pd.period_range(START, END, freq="M")
    rows: list[dict[str, Any]] = []
    txn_id = 1
    for product in PRODUCTS:
        for period in months:
            if product == EMPTY_MONTH_PRODUCT and str(period) == EMPTY_MONTH:
                continue
            date = period.to_timestamp(how="start") + pd.Timedelta(days=4)
            season = _season_of(date.month)
            if product == WRONG_SEASON_PRODUCT and str(period) == WRONG_SEASON_MONTH:
                season = "Summer" if season != "Summer" else "Winter"
            rows.append({
                "TransactionID": f"T{txn_id:05d}",
                "ProductID": f"P{txn_id:05d}",
                "ProductName": product,
                "Category": "Clothing",
                "Price": 29.99,
                "QuantitySold": 10 + (date.month % 5) + (date.year % 3) * 2,
                "SaleDate": date.strftime("%d/%m/%Y"),
                "Season": season,
                "Year": date.year,
            })
            txn_id += 1
    rows.append(dict(rows[-1]))  # planted duplicate TransactionID
    path = tmp_path / "tiny.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def tiny_config(tmp_path: Path, tiny_transactions_csv: Path) -> dict[str, Any]:
    return {
        "seed": 42,
        "season_length": 12,
        "stores": {"tiny": {"name": "Tiny Store", "path": str(tiny_transactions_csv)}},
        "data": {
            "encoding": "utf-8",
            "date_column": "SaleDate",
            "date_format": "%d/%m/%Y",
            "quantity_column": "QuantitySold",
            "start": START,
            "end": END,
        },
        "forecast": {"end": "2024-12"},
        "backtest": {"origins": ["2021-12", "2022-12"], "horizon": 12},
        "models": ["naive", "seasonal_naive", "historical_mean", "drift"],
        "paths": {
            "outputs": str(tmp_path / "outputs"),
            "report": str(tmp_path / "reports" / "results.md"),
        },
    }


@pytest.fixture
def tiny_config_path(tmp_path: Path, tiny_config: dict[str, Any]) -> Path:
    import yaml

    path = tmp_path / "config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(tiny_config, f)
    return path
