"""Panel feature engineering for the global LightGBM model. Lag and rolling
features for a product at time t only ever read values at t-1 or earlier,
so slicing a store's monthly panel to `<= train_end` before calling
`build_panel` is sufficient to keep every row leakage-free -- no separate
leakage flag is needed.

`next_step_row` is the single mechanism recursive multi-step forecasting
uses: lags, rolling mean/std, and the recursion itself all read from one
growing `pd.Series` buffer (history + each step's own prior prediction
appended), so there is no separate "lag updater" vs "rolling updater" code
path to keep in sync.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_LAGS: tuple[int, ...] = (1, 2, 3, 12)
DEFAULT_WINDOWS: tuple[int, ...] = (3, 6, 12)


def month_of_year_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    month = index.month.to_numpy()
    return pd.DataFrame(
        {
            "month": month,
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
        },
        index=index,
    )


def feature_names(lags: Sequence[int] = DEFAULT_LAGS, windows: Sequence[int] = DEFAULT_WINDOWS) -> list[str]:
    names = [f"lag_{lag}" for lag in lags]
    for window in windows:
        names += [f"roll_mean_{window}", f"roll_std_{window}"]
    names += ["month", "month_sin", "month_cos", "product_code"]
    return names


def _product_frame(series: pd.Series, lags: Sequence[int], windows: Sequence[int]) -> pd.DataFrame:
    frame = pd.DataFrame({"target": series})
    for lag in lags:
        frame[f"lag_{lag}"] = series.shift(lag)

    shifted = series.shift(1)  # never let a month's own value leak into its own rolling stats
    for window in windows:
        frame[f"roll_mean_{window}"] = shifted.rolling(window).mean()
        frame[f"roll_std_{window}"] = shifted.rolling(window).std(ddof=1)

    frame = frame.join(month_of_year_features(series.index))
    return frame


def build_panel(
    monthly: pd.DataFrame,
    train_end: pd.Timestamp,
    lags: Sequence[int] = DEFAULT_LAGS,
    windows: Sequence[int] = DEFAULT_WINDOWS,
) -> tuple[pd.DataFrame, list[str]]:
    """Long-format training panel built only from rows `<= train_end`.

    Returns `(panel, categories)` where `categories` is the fixed, sorted
    product list the caller must store and reuse verbatim at predict time
    (see `FittedGlobalModel.categories` in `models/gbm.py`).
    """
    hist = monthly.loc[monthly.index <= train_end]
    categories = sorted(hist.columns)
    category_code = {name: code for code, name in enumerate(categories)}

    frames = []
    for product in categories:
        frame = _product_frame(hist[product], lags, windows)
        frame["product"] = product
        frame["product_code"] = category_code[product]
        frame["date"] = frame.index
        frames.append(frame)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=feature_names(lags, windows)).reset_index(drop=True)
    return panel, categories


def next_step_row(
    history: pd.Series,
    next_date: pd.Timestamp,
    product_code: int,
    lags: Sequence[int] = DEFAULT_LAGS,
    windows: Sequence[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """One feature row for `next_date`, built from `history` -- the growing
    buffer of real values plus whatever recursive predictions have already
    been appended for earlier steps in this forecast. The caller appends
    its own prediction for `next_date` before calling this again for the
    following step.
    """
    row: dict[str, float] = {}
    for lag in lags:
        row[f"lag_{lag}"] = float(history.iloc[-lag]) if len(history) >= lag else float(history.iloc[0])
    for window in windows:
        tail = history.iloc[-window:]
        row[f"roll_mean_{window}"] = float(tail.mean())
        row[f"roll_std_{window}"] = float(tail.std(ddof=1)) if len(tail) >= 2 else 0.0

    moy = month_of_year_features(pd.DatetimeIndex([next_date]))
    row["month"] = int(moy["month"].iloc[0])
    row["month_sin"] = float(moy["month_sin"].iloc[0])
    row["month_cos"] = float(moy["month_cos"].iloc[0])
    row["product_code"] = product_code
    return pd.DataFrame([row])
