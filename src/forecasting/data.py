"""The only module that touches raw transaction files.

Loads, validates and aggregates store transactions into a monthly product x
time matrix. Soft data-quality issues (season/year mismatches, duplicate
transaction IDs, ...) are measured and reported, never silently dropped;
only missing columns, unparseable dates and negative quantities are fatal.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "TransactionID", "ProductName", "Category", "Price",
    "QuantitySold", "SaleDate", "Season", "Year",
]


@dataclass
class DataQualityReport:
    store: str
    rows: int
    products: int
    year_mismatch: float
    season_mismatch: float
    duplicate_transaction_ids: int
    amount_consistency: float | None
    multi_category_products: int
    seasonality_r2_median: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StoreData:
    key: str
    name: str
    monthly: pd.DataFrame
    categories: dict[str, str]
    quality: DataQualityReport


def _season_of(month: pd.Series) -> np.ndarray:
    return np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
        ["Winter", "Spring", "Summer"],
        "Fall",
    )


def load_raw(path: Path, cfg: dict[str, Any]) -> pd.DataFrame:
    data_cfg = cfg["data"]
    df = pd.read_csv(path, encoding=data_cfg["encoding"])
    df[data_cfg["date_column"]] = pd.to_datetime(
        df[data_cfg["date_column"]], format=data_cfg["date_format"], errors="coerce",
    )
    return df


def validate(df: pd.DataFrame, store: str) -> DataQualityReport:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{store}: missing required columns: {missing}")

    if df["SaleDate"].isna().any():
        bad = int(df["SaleDate"].isna().sum())
        raise ValueError(f"{store}: {bad} row(s) have an unparseable SaleDate")

    if (df["QuantitySold"] < 0).any():
        raise ValueError(f"{store}: negative QuantitySold")

    year_mismatch = float((df["Year"] != df["SaleDate"].dt.year).mean())
    season_mismatch = float((df["Season"] != _season_of(df["SaleDate"].dt.month)).mean())
    duplicate_transaction_ids = int(df["TransactionID"].duplicated().sum())
    multi_category_products = int((df.groupby("ProductName")["Category"].nunique() > 1).sum())

    amount_consistency: float | None = None
    if {"DiscountApplied", "TransactionAmount"} <= set(df.columns):
        calc = df["Price"] * df["QuantitySold"] * (1 - df["DiscountApplied"] / 100)
        amount_consistency = float(np.isclose(calc, df["TransactionAmount"], rtol=0.01).mean())

    seasonality_r2_median = _seasonality_r2_median(df)

    if year_mismatch > 0:
        logger.warning("%s: Year disagrees with SaleDate on %.1f%% of rows", store, year_mismatch * 100)
    if season_mismatch > 0:
        logger.warning("%s: Season disagrees with SaleDate on %.1f%% of rows", store, season_mismatch * 100)
    if duplicate_transaction_ids > 0:
        logger.warning("%s: %d duplicate TransactionID values", store, duplicate_transaction_ids)
    if amount_consistency is not None and amount_consistency < 1.0:
        logger.warning("%s: TransactionAmount disagrees with Price*Qty*(1-discount) on %.1f%% of rows",
                        store, (1 - amount_consistency) * 100)
    if multi_category_products > 0:
        logger.warning("%s: %d product(s) have more than one Category", store, multi_category_products)

    return DataQualityReport(
        store=store,
        rows=len(df),
        products=df["ProductName"].nunique(),
        year_mismatch=year_mismatch,
        season_mismatch=season_mismatch,
        duplicate_transaction_ids=duplicate_transaction_ids,
        amount_consistency=amount_consistency,
        multi_category_products=multi_category_products,
        seasonality_r2_median=seasonality_r2_median,
    )


def _seasonality_r2_median(df: pd.DataFrame) -> float:
    monthly = df.groupby(["ProductName", pd.Grouper(key="SaleDate", freq="ME")])["QuantitySold"].sum().unstack(0)
    r2_values = []
    for product in monthly.columns:
        series = monthly[product].dropna()
        if series.var() > 0:
            fitted = series.groupby(series.index.month).transform("mean")
            r2_values.append(1 - ((series - fitted) ** 2).sum() / ((series - series.mean()) ** 2).sum())
    return float(np.median(r2_values)) if r2_values else float("nan")


def to_monthly(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    grouped = df.groupby(["ProductName", pd.Grouper(key="SaleDate", freq="ME")])["QuantitySold"].sum().unstack(0)
    start_period, end_period = pd.Period(start, freq="M"), pd.Period(end, freq="M")
    full_index = pd.period_range(start_period, end_period, freq="M").to_timestamp(how="end").normalize()
    return grouped.reindex(full_index, fill_value=0.0).fillna(0.0)


def product_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _categories_by_product(df: pd.DataFrame) -> dict[str, str]:
    return df.groupby("ProductName")["Category"].agg(lambda s: s.value_counts().idxmax()).to_dict()


def load_store(key: str, cfg: dict[str, Any]) -> StoreData:
    store_cfg = cfg["stores"][key]
    df = load_raw(Path(store_cfg["path"]), cfg)
    quality = validate(df, key)
    monthly = to_monthly(df, cfg["data"]["start"], cfg["data"]["end"])
    categories = _categories_by_product(df)
    return StoreData(key=key, name=store_cfg["name"], monthly=monthly, categories=categories, quality=quality)
