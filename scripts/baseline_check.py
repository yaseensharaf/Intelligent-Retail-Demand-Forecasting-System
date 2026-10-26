"""Reproduce the AUDIT.md findings: data-quality checks + naive baselines vs the reported hybrid.

Run from the repository root:  python scripts/baseline_check.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = {"H&M": "data/H_M.csv", "Retail Store 1": "data/R1(Main).csv", "Retail Store 2": "data/R2.csv"}
METRICS = ROOT / "Sales_Forecast_Results_with_Accuracy" / "H&M_accuracy_metrics.csv"


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / path, encoding="cp1252")
    df["SaleDate"] = pd.to_datetime(df["SaleDate"], format="%d/%m/%Y")
    return df


def monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Products as columns, full month-end calendar, missing months = 0."""
    m = df.groupby(["ProductName", pd.Grouper(key="SaleDate", freq="ME")])["QuantitySold"].sum().unstack(0)
    full = pd.date_range(m.index.min(), m.index.max(), freq="ME")
    return m.reindex(full).fillna(0)


def season_of(month: pd.Series) -> np.ndarray:
    return np.select([month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
                     ["Winter", "Spring", "Summer"], "Fall")


def data_quality(name: str, df: pd.DataFrame) -> dict:
    out = {
        "store": name,
        "rows": len(df),
        "products": df["ProductName"].nunique(),
        "year_mismatch": (df["Year"] != df["SaleDate"].dt.year).mean(),
        "season_mismatch": (df["Season"] != season_of(df["SaleDate"].dt.month)).mean(),
        "dup_txn_ids": int(df["TransactionID"].duplicated().sum()),
        "multi_category_products": int((df.groupby("ProductName")["Category"].nunique() > 1).sum()),
    }
    if {"DiscountApplied", "TransactionAmount"} <= set(df.columns):
        calc = df["Price"] * df["QuantitySold"] * (1 - df["DiscountApplied"] / 100)
        out["amount_consistent"] = np.isclose(calc, df["TransactionAmount"], rtol=0.01).mean()
    ms = monthly(df)
    r2 = []
    for p in ms:
        s = ms[p]
        if s.var() > 0:
            fitted = s.groupby(s.index.month).transform("mean")
            r2.append(1 - ((s - fitted) ** 2).sum() / ((s - s.mean()) ** 2).sum())
    out["seasonality_r2_median"] = float(np.median(r2))
    return out


def mape(a, f):
    return float(np.mean(np.abs(a - f) / a) * 100)


def baselines_hm() -> pd.DataFrame:
    ms = monthly(load(RAW["H&M"]))
    train, test = ms.loc[:"2022"], ms.loc["2023"]
    rows = []
    for p in test:
        a, tr = test[p].to_numpy(), train[p].to_numpy()
        rows.append({
            "Product": p,
            "mean_2015_2022": mape(a, np.full(12, tr.mean())),
            "mean_2022": mape(a, np.full(12, tr[-12:].mean())),
            "seasonal_naive": mape(a, tr[-12:]),
            "mean_2015_2022_first9": mape(a[:9], np.full(9, tr.mean())),
        })
    res = pd.DataFrame(rows)
    if METRICS.exists():
        rep = pd.read_csv(METRICS)[["Product", "MAPE"]].rename(columns={"MAPE": "hybrid_reported"})
        rep["hybrid_reported"] = pd.to_numeric(rep["hybrid_reported"], errors="coerce")
        res = res.merge(rep, on="Product", how="left")
    return res


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    print("=== Data quality ===")
    print(pd.DataFrame([data_quality(n, load(p)) for n, p in RAW.items()]).round(3).to_string(index=False))
    print("\n(Seasonality R^2 ~0.10 is what pure noise gives with 12 month means over 108 points.)")

    print("\n=== H&M 2023 hold-out, mean MAPE over products ===")
    res = baselines_hm()
    print(res.drop(columns="Product").mean().round(2).to_string())
    if "hybrid_reported" in res:
        wins = (res["hybrid_reported"] < res["mean_2015_2022_first9"]).sum()
        print(f"\nHybrid beats the flat historical mean on {wins} of {len(res)} products "
              "(compared on the 9 months the hybrid was actually scored on).")
