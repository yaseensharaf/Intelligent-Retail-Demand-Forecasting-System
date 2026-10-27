"""`python -m forecasting backtest|report|run`"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from forecasting import backtest, data, forecast, report
from forecasting.config import load_config, set_seeds

logger = logging.getLogger(__name__)

ALL_STORES = "all"


def _resolve_stores(store_arg: str | None, cfg: dict) -> list[str]:
    if store_arg is None or store_arg == ALL_STORES:
        return list(cfg["stores"])
    if store_arg not in cfg["stores"]:
        raise ValueError(f"unknown store '{store_arg}'; choices: {ALL_STORES}, {', '.join(cfg['stores'])}")
    return [store_arg]


def _run_backtest(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    set_seeds(cfg["seed"])
    stores = _resolve_stores(args.store, cfg)

    outputs = Path(cfg["paths"]["outputs"])
    outputs.mkdir(parents=True, exist_ok=True)
    quality_path = outputs / "data_quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}

    for store_key in stores:
        start = time.perf_counter()
        store = data.load_store(store_key, cfg)
        forecasts = backtest.backtest_store(store, cfg, products=args.products)
        scores = backtest.score(forecasts, store.monthly, m=cfg["season_length"])

        forecasts.to_csv(outputs / f"backtest_{store_key}_forecasts.csv", index=False)
        scores.to_csv(outputs / f"backtest_{store_key}_scores.csv", index=False)
        quality[store_key] = store.quality.to_dict()

        logger.info("backtest %s: %d forecast rows in %.1fs", store_key, len(forecasts), time.perf_counter() - start)

    quality_path.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    return 0


def _run_report(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    stores = _resolve_stores(args.store, cfg)
    path = report.write_report(cfg, stores)
    logger.info("wrote %s", path)
    return 0


def _run_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    set_seeds(cfg["seed"])
    stores = _resolve_stores(args.store, cfg)
    written = forecast.run_all(stores, cfg, products=args.products)
    for store_key, path in written.items():
        logger.info("forecast %s -> %s", store_key, path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forecasting")
    sub = parser.add_subparsers(dest="command", required=True)

    backtest_parser = sub.add_parser("backtest")
    backtest_parser.add_argument("--store", default=ALL_STORES)
    backtest_parser.add_argument("--config", default="config/default.yaml")
    backtest_parser.add_argument("--products", type=int, default=None,
                                  help="only backtest the first N products of each store (dev speed)")
    backtest_parser.set_defaults(func=_run_backtest)

    report_parser = sub.add_parser("report")
    report_parser.add_argument("--store", default=None)
    report_parser.add_argument("--config", default="config/default.yaml")
    report_parser.set_defaults(func=_run_report)

    run_parser = sub.add_parser("run")
    run_parser.add_argument("--store", default=ALL_STORES)
    run_parser.add_argument("--config", default="config/default.yaml")
    run_parser.add_argument("--products", type=int, default=None,
                             help="only forecast the first N products of each store (dev speed)")
    run_parser.set_defaults(func=_run_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
