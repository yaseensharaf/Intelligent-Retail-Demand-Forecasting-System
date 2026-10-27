"""Rolling-origin backtesting. Every split is enforced to use only data
strictly before the forecast origin -- see test_backtest.py for the leakage
guards this module must keep passing.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd

from forecasting.data import StoreData
from forecasting.metrics import all_metrics, diebold_mariano, interval_coverage
from forecasting.models import build_model
from forecasting.models.base import INTERVAL_COLUMNS
from forecasting.models.gbm import fit_global_lightgbm

METRIC_COLUMNS = ["mae", "rmse", "wape", "bias", "smape", "mape", "mase"]


def _to_month_end(origin: str | pd.Timestamp) -> pd.Timestamp:
    if isinstance(origin, str):
        return pd.Period(origin, freq="M").to_timestamp(how="end").normalize()
    return pd.Timestamp(origin)


def split_at(y: pd.Series, origin: str | pd.Timestamp, h: int) -> tuple[pd.Series, pd.Series]:
    origin_ts = _to_month_end(origin)
    train = y[y.index <= origin_ts]
    test = y[y.index > origin_ts].iloc[:h]

    if len(train) == 0:
        raise ValueError(f"no training data at or before origin {origin_ts}")
    if len(test) != h:
        raise ValueError(f"expected {h} test months after {origin_ts}, found {len(test)}")
    if train.index.max() >= test.index.min():
        raise ValueError("training data leaks into the test window")

    return train, test


def backtest_series(
    product: str,
    y: pd.Series,
    model_names: list[str],
    origins: list[str],
    h: int,
    model_params: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    model_params = model_params or {}
    rows: list[dict[str, Any]] = []
    for origin in origins:
        train, test = split_at(y, origin, h)
        origin_ts = _to_month_end(origin)
        for name in model_names:
            model = build_model(name, **model_params.get(name, {})).fit(train)
            forecast = model.predict(h)
            for step, (date, actual) in enumerate(test.items(), start=1):
                f_row = forecast.loc[date]
                rows.append({
                    "product": product,
                    "model": name,
                    "origin": origin_ts,
                    "date": date,
                    "step": step,
                    "actual": float(actual),
                    **{col: float(f_row[col]) for col in INTERVAL_COLUMNS},
                })
    return pd.DataFrame(rows)


def backtest_global(
    store: StoreData, model_name: str, origins: list[str], h: int, params: dict[str, Any],
) -> pd.DataFrame:
    """Sibling to `backtest_series` for models trained once across the
    whole panel per origin (currently just `lightgbm_global`) rather than
    per product -- see `models/gbm.py` for why this can't go through the
    single-series `Forecaster` seam. Reuses `split_at` per product for the
    same leakage guard every other model gets.
    """
    rows: list[dict[str, Any]] = []
    for origin in origins:
        origin_ts = _to_month_end(origin)
        fitted = fit_global_lightgbm(store.monthly, origin_ts, params)
        for product in store.monthly.columns:
            train, test = split_at(store.monthly[product], origin, h)
            forecast = fitted.predict_product(product, train, h)
            for step, (date, actual) in enumerate(test.items(), start=1):
                f_row = forecast.loc[date]
                rows.append({
                    "product": product,
                    "model": model_name,
                    "origin": origin_ts,
                    "date": date,
                    "step": step,
                    "actual": float(actual),
                    **{col: float(f_row[col]) for col in INTERVAL_COLUMNS},
                })
    return pd.DataFrame(rows)


def backtest_store(store: StoreData, cfg: dict[str, Any], products: int | None = None) -> pd.DataFrame:
    origins = cfg["backtest"]["origins"]
    h = cfg["backtest"]["horizon"]
    model_names = cfg["models"]
    model_params = cfg.get("model_params", {})

    columns = list(store.monthly.columns[:products]) if products else list(store.monthly.columns)
    frames = [
        backtest_series(product, store.monthly[product], model_names, origins, h, model_params)
        for product in columns
    ]

    global_names = cfg.get("global_models", [])
    if global_names:
        global_params = cfg.get("global_model_params", {})
        restricted_store = store if products is None else replace(store, monthly=store.monthly[columns])
        for name in global_names:
            frames.append(backtest_global(restricted_store, name, origins, h, global_params.get(name, {})))

    return pd.concat(frames, ignore_index=True)


def score(forecasts: pd.DataFrame, monthly: pd.DataFrame, m: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (product, model, origin), group in forecasts.groupby(["product", "model", "origin"]):
        group = group.sort_values("step")
        y = group["actual"].to_numpy()
        f = group["mean"].to_numpy()
        y_train = monthly.loc[monthly.index <= origin, product].to_numpy()
        row = {
            "product": product,
            "model": model,
            "origin": origin,
            **all_metrics(y, f, y_train, m=m),
            "coverage80": interval_coverage(y, group["lo80"].to_numpy(), group["hi80"].to_numpy()),
            "coverage95": interval_coverage(y, group["lo95"].to_numpy(), group["hi95"].to_numpy()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _pick_winner(group: pd.DataFrame) -> str:
    candidate = group["mase"]
    if candidate.isna().all():
        candidate = group["wape"]
    return group.loc[candidate.idxmin(), "model"]


def _win_counts(scores_df: pd.DataFrame) -> pd.Series:
    winners = scores_df.groupby(["product", "origin"]).apply(_pick_winner, include_groups=False)
    counts = winners.value_counts()
    return counts.reindex(scores_df["model"].unique(), fill_value=0).astype(int)


def summarise(scores_df: pd.DataFrame) -> pd.DataFrame:
    stat_cols = [*METRIC_COLUMNS, "coverage80", "coverage95"]
    agg = scores_df.groupby("model")[stat_cols].agg(["mean", "median"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index()
    wins = _win_counts(scores_df)
    # one (product, origin) pair is "won" by exactly one model each, so this
    # column counts product x origin pairs, not products.
    agg["wins_product_origin_pairs"] = agg["model"].map(wins)
    return agg


BASELINE_MODELS = ["naive", "seasonal_naive", "historical_mean", "drift"]


def pick_reference_baseline(scores_df: pd.DataFrame, baseline_names: list[str] | None = None) -> str:
    """Whichever of the baseline models has the best (lowest) mean MASE
    for this store -- falling back to WAPE if MASE is NaN for all of
    them -- per ROADMAP's "DM-test p-value vs best baseline" (the earlier
    hardcoded `reference="historical_mean"` wasn't necessarily the best
    baseline for every store).
    """
    baseline_names = baseline_names if baseline_names is not None else BASELINE_MODELS
    present = [name for name in baseline_names if name in set(scores_df["model"])]
    if not present:
        raise ValueError("none of the configured baseline models are present in scores_df")
    summary = summarise(scores_df[scores_df["model"].isin(present)])
    candidate = summary["mase_mean"]
    if candidate.isna().all():
        candidate = summary["wape_mean"]
    return str(summary.loc[candidate.idxmin(), "model"])


def compare_to_reference(forecasts: pd.DataFrame, reference: str = "historical_mean") -> pd.DataFrame:
    h = int(forecasts["step"].max())
    ref = forecasts[forecasts["model"] == reference]
    rows: list[dict[str, Any]] = []
    for model_name in forecasts["model"].unique():
        if model_name == reference:
            continue
        cur = forecasts[forecasts["model"] == model_name]
        merged = cur.merge(ref, on=["product", "origin", "step"], suffixes=("", "_ref"))
        e1 = (merged["actual"] - merged["mean"]).to_numpy()
        e2 = (merged["actual"] - merged["mean_ref"]).to_numpy()
        stat, p_value = diebold_mariano(e1, e2, h=h)
        rows.append({"model": model_name, "reference": reference, "dm_stat": stat, "p_value": p_value})
    return pd.DataFrame(rows)
