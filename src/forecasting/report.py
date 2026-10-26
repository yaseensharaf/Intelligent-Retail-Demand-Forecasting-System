"""Render reports/results.md. Reads only from outputs/ (written by the
`backtest` CLI command) -- never touches data/raw/ or recomputes forecasts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from forecasting import backtest

# AUDIT.md section 1.5: the legacy ARIMA-LSTM hybrid's reported H&M MAPE.
# That figure was scored on Jan-Sep 2023 only (9 of 12 months), not the full
# year the project's README claims -- kept here only as a historical anchor.
LEGACY_HYBRID_MAPE = 15.2
LEGACY_HYBRID_NOTE = "scored on Jan-Sep 2023 only (9 of 12 months), H&M"

LOWER_IS_BETTER = ["mae", "rmse", "wape", "mase", "smape", "mape"]


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        cells = ["" if pd.isna(v) else str(v) for v in row]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _bold_best(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.astype(object).copy()
    for col in columns:
        if col not in df.columns:
            continue
        values = df[col]
        if values.notna().any():
            best_idx = values.idxmin()
            out.loc[best_idx, col] = f"**{values.loc[best_idx]:.3f}**"
    return out


def _data_quality_table(quality: dict[str, dict[str, Any]], stores: list[str]) -> str:
    rows = pd.DataFrame([quality[s] for s in stores if s in quality])
    return _dataframe_to_markdown(rows.round(3))


def _metrics_table(scores: pd.DataFrame) -> str:
    summary = backtest.summarise(scores)
    display_cols = (
        ["model"]
        + [f"{m}_mean" for m in backtest.METRIC_COLUMNS]
        + ["coverage80_mean", "coverage95_mean", "wins"]
    )
    table = summary[display_cols].round(3)
    table = _bold_best(table, [f"{m}_mean" for m in LOWER_IS_BETTER])
    return _dataframe_to_markdown(table)


def _comparison_table(forecasts: pd.DataFrame, reference: str) -> str:
    comparison = backtest.compare_to_reference(forecasts, reference=reference)
    return _dataframe_to_markdown(comparison.round(4))


def write_report(cfg: dict[str, Any], stores: list[str]) -> Path:
    outputs = Path(cfg["paths"]["outputs"])
    report_path = Path(cfg["paths"]["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)

    quality_path = outputs / "data_quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}

    lines = ["# Results", ""]
    lines.append("## Data diagnostics")
    lines.append("")
    lines.append(_data_quality_table(quality, stores))
    lines.append("")

    for store in stores:
        store_name = cfg["stores"][store]["name"]
        lines.append(f"## {store_name}")
        lines.append("")

        scores = pd.read_csv(outputs / f"backtest_{store}_scores.csv")
        forecasts = pd.read_csv(outputs / f"backtest_{store}_forecasts.csv", parse_dates=["origin", "date"])

        lines.append("### Metrics (mean over products & origins, best in bold)")
        lines.append("")
        lines.append(_metrics_table(scores))
        lines.append("")

        lines.append("### Win counts and Diebold-Mariano vs historical_mean")
        lines.append("")
        lines.append(_comparison_table(forecasts, reference="historical_mean"))
        lines.append("")

    lines.append("## Reference: legacy hybrid")
    lines.append("")
    lines.append(f"The notebook-era ARIMA-LSTM hybrid reported **{LEGACY_HYBRID_MAPE}% MAPE** "
                  f"({LEGACY_HYBRID_NOTE}; see AUDIT.md). It is not reproduced by this pipeline.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
