# Phase 1 plan — reproducible data + evaluation framework

Branch: `phase-1-pipeline` (from `main` after Phase 0 is merged).
Scope: baselines and evaluation only. No ARIMA/LSTM yet (Phase 2). On this data the
historical-mean baseline will probably win — that is the honest result to report.

## Files to create
```
pyproject.toml
config/default.yaml
src/forecasting/__init__.py
src/forecasting/__main__.py
src/forecasting/config.py
src/forecasting/data.py
src/forecasting/metrics.py
src/forecasting/models/__init__.py
src/forecasting/models/base.py
src/forecasting/models/baselines.py
src/forecasting/backtest.py
src/forecasting/report.py
src/forecasting/cli.py
tests/conftest.py
tests/test_data.py
tests/test_metrics.py
tests/test_models.py
tests/test_backtest.py
tests/test_cli.py
```
`src/app.py` and `notebooks/legacy/` stay untouched until Phase 4. Keep the existing `ruff.toml`
(or move its settings into `pyproject.toml`, not both).

## pyproject.toml
Makes `src/forecasting` installable (`pip install -e .`). Pytest settings: `testpaths = ["tests"]`,
`pythonpath = ["src"]`. Closes AUDIT §5 (packaging).

## config/default.yaml
```yaml
seed: 42
season_length: 12
stores:
  hm:     {name: "H&M",            path: data/raw/hm.csv}
  store1: {name: "Retail Store 1", path: data/raw/store1.csv}
  store2: {name: "Retail Store 2", path: data/raw/store2.csv}
data:
  encoding: utf-8
  date_column: SaleDate
  date_format: "%d/%m/%Y"
  quantity_column: QuantitySold
  start: "2015-01"
  end:   "2023-12"
backtest:
  origins: ["2020-12", "2021-12", "2022-12"]   # train up to and including this month
  horizon: 12
models: [naive, seasonal_naive, historical_mean, drift]
paths:
  outputs: outputs
  report:  reports/results.md
```
Closes §3 (hard-coded paths and weights spread across notebooks).

## config.py
- `load_config(path: str | Path = "config/default.yaml") -> dict`
- `set_seeds(seed: int) -> None` — seeds `random`, `numpy` (Phase 2 adds TensorFlow).

Closes §1.5 (no seeds).

## data.py
The only code that touches raw files.
```python
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
    def to_dict(self) -> dict: ...

@dataclass
class StoreData:
    key: str
    name: str
    monthly: pd.DataFrame        # index month-end (freq "ME"), columns product names
    categories: dict[str, str]   # product -> most common category
    quality: DataQualityReport

def load_raw(path: Path, cfg: dict) -> pd.DataFrame             # explicit encoding + date format
def validate(df: pd.DataFrame, store: str) -> DataQualityReport  # warn on soft issues; raise on missing columns, bad dates, negative qty
def to_monthly(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame   # full calendar, empty months = 0
def product_slug(name: str) -> str                              # "Lace-Up Boots" -> "lace-up-boots"
def load_store(key: str, cfg: dict) -> StoreData
```
Closes §2 (issues measured and reported; gaps filled with 0; explicit encoding),
§3 (one loader, `freq="ME"`), §4 (slugs instead of lowercasing).

## metrics.py
Floats in, no rounding before scoring.
```python
def mae(y, f) -> float
def rmse(y, f) -> float
def wape(y, f) -> float
def bias(y, f) -> float                       # mean(f - y); positive = over-forecast
def smape(y, f) -> float
def mape(y, f) -> float                       # NaN if any actual is 0; kept for comparison with old results
def mase(y, f, y_train, m=12) -> float
def interval_coverage(y, lo, hi) -> float
def diebold_mariano(e1, e2, h=1) -> tuple[float, float]   # (stat, p) with Harvey–Leybourne–Newbold correction
def all_metrics(y, f, y_train, m=12) -> dict[str, float]
```
Closes §1.5 (MAPE only, no significance test, rounding before scoring).

## models/base.py
```python
INTERVAL_COLUMNS = ["mean", "lo80", "hi80", "lo95", "hi95"]

class Forecaster(Protocol):
    name: str
    def fit(self, y: pd.Series) -> "Forecaster": ...
    def predict(self, h: int) -> pd.DataFrame: ...

def future_index(last_date: pd.Timestamp, h: int) -> pd.DatetimeIndex   # starts the month after last_date
def normal_intervals(mean: np.ndarray, sigma: np.ndarray) -> pd.DataFrame
```
`future_index` is the single place forecast dates are created. Closes §1.4.

## models/baselines.py
Intervals use the formulas in Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (citable).

| Class | Forecast | Interval std |
|---|---|---|
| `NaiveForecaster` | last value | σ·√h |
| `SeasonalNaiveForecaster(m=12)` | same month last year | σ·√(⌊(h−1)/m⌋+1) |
| `HistoricalMeanForecaster(window=None)` | mean of history (or last `window` months) | σ·√(1+1/T) |
| `DriftForecaster` | line from first to last value | σ·√(h(1+h/T)) |

Registry `MODELS = {"naive": ..., ...}` and `build_model(name: str, **params) -> Forecaster`.
Closes §1.5 (no baselines).

## backtest.py
```python
def split_at(y: pd.Series, origin: pd.Timestamp, h: int) -> tuple[pd.Series, pd.Series]
    # train = y[y.index <= origin]; test = next h months; assert train max < test min, len(test) == h
def backtest_series(product: str, y: pd.Series, model_names: list[str], origins: list[str], h: int) -> pd.DataFrame
    # rows: product, model, origin, date, step, actual, mean, lo80, hi80, lo95, hi95
def backtest_store(store: StoreData, cfg: dict) -> pd.DataFrame
def score(forecasts: pd.DataFrame, monthly: pd.DataFrame, m: int) -> pd.DataFrame
    # per product x model x origin: MAE, RMSE, WAPE, MASE, bias, sMAPE, MAPE, coverage80, coverage95
def summarise(scores: pd.DataFrame) -> pd.DataFrame          # per model mean/median + product win counts
def compare_to_reference(forecasts: pd.DataFrame, reference: str = "historical_mean") -> pd.DataFrame
```
Outputs (git-ignored `outputs/`): `backtest_<store>_forecasts.csv`, `backtest_<store>_scores.csv`,
`data_quality.json`. Closes §1.5 (single split, 9 months, Stores 1–2 unevaluated), §3 (reproducibility).

## report.py
`write_report(cfg: dict, stores: list[str]) -> Path` builds `reports/results.md` only from `outputs/`:
1. Data diagnostics table per store.
2. Metrics table per store (models × metrics, best in bold).
3. Win counts and Diebold–Mariano p-values.
4. Reference line for the legacy hybrid's reported 15.2% MAPE, labelled as scored on Jan–Sep 2023 only.

## cli.py / __main__.py
```
python -m forecasting backtest --store all|hm|store1|store2 [--config PATH]
python -m forecasting report [--config PATH]
python -m forecasting run        # prints "added in Phase 2", exits 0
```
`main(argv: list[str] | None = None) -> int`. Logs run time per store.

## Tests
- **conftest.py**: tiny synthetic transactions CSV (3 products, 2019–2023, one empty month, one wrong
  season label, one duplicate ID) and a config in `tmp_path` — tests never touch real data.
- **test_data.py**: monthly index equals `pd.date_range(start, end, freq="ME")` with the empty month = 0;
  `validate` flags the planted issues; missing column raises; `product_slug` stable, URL-safe, unique.
- **test_metrics.py**: hand-computed values; MAPE NaN on zero actual; MASE == 1 when forecast is the
  seasonal naive; DM p≈1 for identical errors, p<0.05 for clearly better errors.
- **test_models.py** (parametrised over all baselines): 12 rows, correct columns; first forecast date is
  the month after the last training month; lo95 ≤ lo80 ≤ mean ≤ hi80 ≤ hi95; seasonal naive repeats last
  year; historical mean is flat.
- **test_backtest.py** (leakage):
  1. Spy model records the last training date — must be ≤ origin and < first test date for every origin.
  2. Poisoned future — set all values after the origin to 1e6; forecasts must be identical to the clean run.
  3. Each origin yields exactly 12 consecutive test months.
- **test_cli.py**: backtest + report on the fixture config create the output files and a `results.md`
  naming all four baselines. Optional real-data check (skip if `data/raw/hm.csv` missing): H&M historical
  mean on 2023 ≈ 12.03% MAPE, matching `scripts/baseline_check.py`.

## AUDIT items closed
| AUDIT | Closed by |
|---|---|
| §1.4 forecast dates | `future_index` + date test |
| §1.5 baselines, single split, 9 months, MAPE only, rounding, seeds, Stores 1–2 | baselines, backtest, metrics, config |
| §2 encoding, gaps, hidden data issues | `data.py` + report diagnostics |
| §3 copy-paste, reproducibility, `"M"` | package + CLI + config |
| §5 no tests, hand-typed metrics | `tests/`, generated `results.md` |

Not in Phase 1: §1.1–1.3 hybrid (Phase 2), §4 dashboard (Phase 4), CI/Docker (Phase 5).

## Commands at the end
```
pip install -r requirements.txt
pip install -e .
ruff check .
pytest -q
python -m forecasting backtest --store all
python -m forecasting report
git add -A
git commit -m "Phase 1: data pipeline, baselines, rolling-origin backtest"
```
Done when pytest and ruff pass, `reports/results.md` has a table for all three stores, and the H&M
historical-mean figure matches `scripts/baseline_check.py`.
