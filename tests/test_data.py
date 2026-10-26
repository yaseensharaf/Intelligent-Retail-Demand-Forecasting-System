from __future__ import annotations

import pandas as pd
import pytest

from forecasting import data
from tests.conftest import EMPTY_MONTH, EMPTY_MONTH_PRODUCT, END, START, WRONG_SEASON_PRODUCT


def test_to_monthly_covers_full_calendar(tiny_config, tiny_transactions_csv):
    df = data.load_raw(tiny_transactions_csv, tiny_config)
    monthly = data.to_monthly(df, START, END)

    expected_index = pd.date_range("2019-01-31", "2023-12-31", freq="ME")
    pd.testing.assert_index_equal(monthly.index, expected_index)
    assert len(monthly.index) == 60


def test_to_monthly_fills_empty_month_with_zero(tiny_config, tiny_transactions_csv):
    df = data.load_raw(tiny_transactions_csv, tiny_config)
    monthly = data.to_monthly(df, START, END)

    empty_date = pd.Period(EMPTY_MONTH, freq="M").to_timestamp(how="end").normalize()
    assert monthly.loc[empty_date, EMPTY_MONTH_PRODUCT] == 0.0


def test_validate_flags_planted_issues(tiny_config, tiny_transactions_csv):
    df = data.load_raw(tiny_transactions_csv, tiny_config)
    report = data.validate(df, "tiny")

    assert report.duplicate_transaction_ids >= 1
    assert report.season_mismatch > 0
    assert report.amount_consistency is None  # no Discount/TransactionAmount columns
    assert report.rows == len(df)
    assert report.products == len(set(df["ProductName"]))


def test_validate_raises_on_missing_column(tiny_config, tiny_transactions_csv):
    df = data.load_raw(tiny_transactions_csv, tiny_config).drop(columns=["Season"])
    with pytest.raises(ValueError, match="missing required columns"):
        data.validate(df, "tiny")


def test_validate_raises_on_negative_quantity(tiny_config, tiny_transactions_csv):
    df = data.load_raw(tiny_transactions_csv, tiny_config)
    df.loc[0, "QuantitySold"] = -5
    with pytest.raises(ValueError, match="negative QuantitySold"):
        data.validate(df, "tiny")


def test_validate_raises_on_unparseable_date(tiny_config, tiny_transactions_csv):
    df = data.load_raw(tiny_transactions_csv, tiny_config)
    df.loc[0, "SaleDate"] = pd.NaT
    with pytest.raises(ValueError, match="unparseable SaleDate"):
        data.validate(df, "tiny")


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Lace-Up Boots", "lace-up-boots"),
        ("Ties", "ties"),
        ("4K TV", "4k-tv"),
    ],
)
def test_product_slug_known_cases(name, expected):
    assert data.product_slug(name) == expected


def test_product_slug_stable_url_safe_and_unique():
    names = [
        "Lace-Up Boots", "Cotton Blazer", "Wool Trousers", "Levi's 501 Original Fit Jeans",
        "4K TV", "Bowers & Wilkins PX7 S2 Headphones", "De'Longhi Dinamica Espresso Machine",
    ]
    slugs = [data.product_slug(n) for n in names]

    assert slugs == [data.product_slug(n) for n in names]  # stable across calls
    assert len(set(slugs)) == len(slugs)  # unique
    for slug in slugs:
        assert slug == slug.lower()
        assert " " not in slug
        assert all(c.isalnum() or c == "-" for c in slug)


def test_load_store(tiny_config):
    store = data.load_store("tiny", tiny_config)

    assert store.key == "tiny"
    assert store.name == "Tiny Store"
    assert set(store.monthly.columns) == set(["Lace-Up Boots", "Cotton Blazer", "Wool Trousers"])
    assert store.categories["Cotton Blazer"] == "Clothing"
    assert store.quality.store == "tiny"
    assert WRONG_SEASON_PRODUCT in store.categories
