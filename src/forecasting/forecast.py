"""Refit each product's best backtest model on ALL data through the
configured data end-date, and forecast forward to the configured forecast
end-date, correctly aligned via `future_index` -- no repeat of AUDIT.md
section 1.4's year-shift bug, where the dashboard displayed 2023
predictions labelled as 2024.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from forecasting import data
from forecasting.models import build_model
from forecasting.models.gbm import fit_global_lightgbm

logger = logging.getLogger(__name__)

GLOBAL_MODEL_NAME = "lightgbm_global"


def _horizon(cfg: dict[str, Any]) -> int:
    start_period = pd.Period(cfg["data"]["end"], freq="M")
    end_period = pd.Period(cfg["forecast"]["end"], freq="M")
    return (end_period - start_period).n


def _best_model_per_product(scores: pd.DataFrame) -> dict[str, str]:
    """Best model per product, aggregated across every backtest origin --
    same MASE-first, WAPE-fallback tie-break `backtest._pick_winner` uses
    for a single origin, generalised across origins by averaging.
    """
    best: dict[str, str] = {}
    for product, group in scores.groupby("product"):
        per_model = group.groupby("model")[["mase", "wape"]].mean()
        candidate = per_model["mase"]
        if candidate.isna().all():
            candidate = per_model["wape"]
        best[product] = str(candidate.idxmin())
    return best


def forecast_store(store_key: str, cfg: dict[str, Any], products: int | None = None) -> pd.DataFrame:
    store = data.load_store(store_key, cfg)
    scores_path = Path(cfg["paths"]["outputs"]) / f"backtest_{store_key}_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"no backtest scores for {store_key} at {scores_path}; run `backtest` first")
    scores = pd.read_csv(scores_path)

    columns = list(store.monthly.columns[:products]) if products else list(store.monthly.columns)
    best_model = _best_model_per_product(scores)
    h = _horizon(cfg)
    model_params = cfg.get("model_params", {})
    train_end = store.monthly.index[-1]

    # The global model is refit ONCE on the full panel (not just whichever
    # products picked it as their winner) -- that's how it was evaluated
    # during backtesting, and training it on a smaller subset would starve
    # it of exactly the cross-product signal that makes it "global".
    global_products = [p for p in columns if best_model.get(p) == GLOBAL_MODEL_NAME]
    global_fitted = None
    if global_products:
        global_params = cfg.get("global_model_params", {}).get(GLOBAL_MODEL_NAME, {})
        global_fitted = fit_global_lightgbm(store.monthly, train_end, global_params)

    rows: list[dict[str, Any]] = []
    for product in columns:
        name = best_model.get(product)
        if name is None:
            logger.warning("no backtest score for product %r in %s; skipping forecast", product, store_key)
            continue

        if name == GLOBAL_MODEL_NAME:
            forecast = global_fitted.predict_product(product, store.monthly[product], h)
        else:
            model = build_model(name, **model_params.get(name, {})).fit(store.monthly[product])
            forecast = model.predict(h)

        for date, f_row in forecast.iterrows():
            rows.append({
                "product": product,
                "date": date,
                "model_used": name,
                "mean": float(f_row["mean"]),
                "lo80": float(f_row["lo80"]),
                "hi80": float(f_row["hi80"]),
                "lo95": float(f_row["lo95"]),
                "hi95": float(f_row["hi95"]),
            })
    return pd.DataFrame(rows)


def run_all(stores: list[str], cfg: dict[str, Any], products: int | None = None) -> dict[str, Path]:
    outputs = Path(cfg["paths"]["outputs"])
    outputs.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for store_key in stores:
        forecast = forecast_store(store_key, cfg, products)
        path = outputs / f"forecast_{store_key}.csv"
        forecast.to_csv(path, index=False)
        written[store_key] = path
        logger.info("forecast %s: %d rows -> %s", store_key, len(forecast), path)
    return written
