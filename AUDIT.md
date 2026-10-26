# Technical Audit — Intelligent Retail Demand Forecasting System

Audited: 26 Oct 2026, commit on `main` (shallow clone). Every claim below was checked
against the code or reproduced from the data in `data/`. Reproduce the numbers with
`python scripts/baseline_check.py`.

---

## TL;DR

1. **The hybrid model loses to a flat line.** Forecasting every 2023 month as the
   average monthly sales of 2015–2022 gives **12.0% MAPE**; the ARIMA–LSTM hybrid
   gives **15.2%**. Scored on the same 9 months, the hybrid beats that flat baseline on
   only **13 of 80** products.
2. **The LSTM half of the hybrid is not doing what the README says.** Residuals are
   computed between mismatched periods, the "test predictions" are in-sample fits of
   training windows, and the 2024–28 forecast feeds ARIMA *forecasts* (not residuals)
   into the LSTM.
3. **The H&M 2024–2028 forecast is actually the 2023 forecast relabelled.** ARIMA is
   fit on data ending Dec 2022, so step 1 of its forecast is Jan 2023 — but the dates
   are generated starting Jan 2024.
4. **Accuracy is reported over 9 months, not 12.** All 80 products hit the
   "Only 9 sequences available" branch, so metrics cover Jan–Sep 2023.
5. **The data has no learnable seasonal signal.** Month-of-year explains ~13% / 8% / 9%
   of variance (H&M / Store 1 / Store 2) — about what 12 random group means explain in
   108 points of noise (~10%). Several internal consistency checks fail (below).
   It behaves like randomly generated data.
6. **The dashboard has real bugs** (H&M chart images 404 on Linux/macOS-case-sensitive
   setups, Store 1 categories missing, encoding crash on Store 2, cross-store sums) and
   re-reads a 674k-row CSV on every page load.

The good news: all of this is fixable, and fixing it *properly* turns the project
into a much stronger piece of work — an honest, well-evaluated forecasting system.

---

## 1. Forecasting methodology (severity: critical)

### 1.1 Residuals are not residuals
`H_M.ipynb`:
```python
residuals = train.values[-12:] - arima_forecast[:12]
```
`train.values[-12:]` is **2022 actuals**; `arima_forecast` is the **2023 forecast**.
Subtracting them pairs Jan-2022 with Jan-2023 and so on — that's not a model error, it's
year-over-year noise. Correct in-sample residuals are `arima_results.resid`
(or `train - arima_results.fittedvalues`).

`R1.ipynb` / `R2.ipynb` do the reverse and worse:
```python
residuals = arima_forecast.values - train[-60:].values   # 2024–28 forecast minus 2019–23 actuals
```
The sign is also flipped relative to H&M, so the "correction" is added in the opposite
direction between stores.

### 1.2 The LSTM never forecasts
```python
X_pred = X_train[-test_forecast_steps:]
lstm_pred = lstm_model.predict(X_pred)
```
This predicts the model's own **training targets**. Nothing is forecast into 2023. The
LSTM output is then blended with ARIMA month-by-month even though the months don't
correspond.

### 1.3 Future forecast feeds forecasts into the residual model
```python
residuals_future = arima_forecast_future.to_numpy()          # this is the forecast itself
residuals_scaled_future = scaler.fit_transform(...)          # scaler re-fitted on new data
```
The scaler is re-fitted on the forecast, the LSTM is run over it, and the output is
inverse-transformed into the forecast's own scale. So the "LSTM" term is roughly a
smoothed copy of the ARIMA forecast, and `0.9·ARIMA + 0.1·LSTM ≈ ARIMA`. There is also
a 3-step misalignment: LSTM outputs correspond to steps 4…60 but are added to ARIMA
steps 1…57.

### 1.4 H&M forecast dates are shifted by a year
ARIMA is trained to Dec 2022, so `forecast(steps=60)` covers **Jan 2023 – Dec 2027**.
Dates come from `pd.date_range(start=monthly_sales.index[-1], …)` = Jan 2024 onward. The
dashboard therefore shows 2023 predictions labelled as 2024, never refit on 2023 data.

### 1.5 Evaluation weaknesses
- Only 9 test months are scored (see TL;DR #4), but the README says 12.
- A single train/test split; no rolling-origin backtest, so one unlucky year decides everything.
- No baselines at all. Measured on the same split:

  | Method (H&M, 2023, mean over 80 products) | MAPE |
  |---|---|
  | Flat mean of 2015–2022 | **12.0%** |
  | Flat mean of 2022 | 12.4% |
  | ARIMA–LSTM hybrid (reported) | 15.2% |
  | Seasonal naive (same month last year) | 15.8% |

- MAPE alone is fragile. Add MASE (scale-free, baseline-relative), WAPE (what retailers
  actually use), and bias. Test significance with Diebold–Mariano.
- Forecasts are rounded and floored at 1 *before* scoring, adding bias.
- Blend weights (0.90/0.10, 0.95/0.05) are fixed by hand, not tuned on validation data.
- Stores 1 and 2 have no evaluation at all.
- No random seeds → LSTM results aren't reproducible.

## 2. Data quality (severity: critical for the write-up)

| Check | H&M | Store 1 | Store 2 |
|---|---|---|---|
| `Year` ≠ year of `SaleDate` | 0% | 0% | **88.6%** |
| `Season` inconsistent with month | 0% | **74.9%** | **74.9%** |
| Duplicate `TransactionID` | 0 | **21,575** | **11,836** |
| `TransactionAmount` = Price×Qty×(1−discount) | n/a | **3.3%** | 52.8% |
| Products with >1 `ProductID` | 80/80 | 54/54 | 22/40 |
| Products with >1 `Category` | 2 | 0 | 0 |
| Seasonality R² (month-of-year) | 0.13 | 0.08 | 0.09 |

Every product has a different ProductID on almost every row, IDs repeat across
transactions, and derived columns don't agree with their inputs. This is consistent with
a synthetic generator that sampled columns independently. On such data **no model can beat
the mean by much** — which is exactly what §1.5 shows. That's not your fault, but the
dissertation must say it, and the improved version should add a real dataset.

Other data issues:
- The generated Store 2 dashboard file `data/ALLRetail_Store_2.csv` is Windows-1252 encoded
  (byte 0x92 apostrophe); `pd.read_csv` with UTF-8 crashes on it. The raw `R2.csv` itself is valid
  UTF-8. The app reads every file as `ISO-8859-1`, which hides the mismatch.
- Monthly series are built with `groupby` in the merge cells, which drops empty months
  instead of filling 0 (Store 2's dashboard series has gaps, e.g. 2015-05 → 2017-12).
- `data/` holds five near-duplicate copies (`H_M copy.csv`, `All_H&M.csv`, `M(H&M).csv`…).

## 3. Pipeline engineering (severity: high)

- Three notebooks copy-paste the same pipeline with small divergences (that's how the
  sign flip in §1.1 happened). One parameterised module should serve all stores.
- `R1.ipynb` cell 1 can't run: it selects `forecast_data['SaleDate']` and
  `['ProductName']`, neither of which exists. Cell 2 reads `Retail_Store_1_Merged`, but
  cell 1 writes `Retail_Store_1_Merged2`. The committed `ALLRetail_Store_1.csv` came from
  some earlier code that isn't in the repo → **results are not reproducible**.
- Notebooks write `.xlsx`; the app reads `.csv`. The conversion step is manual.
- `create_lstm_data` is redefined inside the product loop.
- `resample('M')` / `freq='M'` are deprecated in pandas ≥ 2.2 (use `'ME'`).
- `auto_arima(..., trace=True)` floods 80× output into a 10 MB notebook.
- Plots saved at `dpi=300` → ~61 MB of PNGs, stored **twice** (`static/` and `src/static/`).
- ARIMA is re-fitted with `statsmodels` after `pmdarima` has already fitted the same model.

## 4. Dashboard (severity: medium–high)

**Bugs**
- `get_hm_products()` lowercases names (`"hat"`), then `/product_image/hat` looks for
  `hat.png`; files are `Hat.png`. Works on Windows only. Same for `/product_sales`, which
  matches case-sensitively.
- Store 1 categories: code reads `data/R1(main).csv`, file is `R1(Main).csv`.
- `/product_sales/<name>` sums the same product name across stores into one series.
- `recommend()` parses dates with `dayfirst=True`; other routes don't. Inconsistent.
- "Trending" sums 2023 actuals **plus five years of forecasts**. That ranks big products,
  not trending ones. Trending should be growth: next-12-month forecast vs last 12 months.
- `/updateAll` is unused by any page, unauthenticated, and writes arbitrary JSON to disk.
- Secret key `'123456'` hard-coded; `debug=True` (Werkzeug debugger = remote code exec if exposed).
- Two dark-mode systems (server session + localStorage) that don't talk to each other.

**Design / performance**
- Every request re-reads CSVs; `/` and `/all_products.html` parse the 674k-row `H_M.csv`
  just to look up categories.
- Six templates of ~550 lines each, mostly duplicated layout and CSS (no `base.html`).
- Charts are static PNGs — no hover, zoom, store comparison or prediction intervals.
- Socket.IO is loaded on every page for an event nothing triggers.
- No accuracy, uncertainty, or actionable output (e.g. reorder quantity) shown to the user.

## 5. Repository hygiene (severity: medium)

- No `requirements.txt`, `.gitignore`, tests, CI, or Docker.
- `src/tempCodeRunnerFile.py` is a VS Code scratch file — delete it.
- Repo is ~250 MB, mostly generated artefacts. Generated outputs should be rebuilt by the
  pipeline, not committed (or kept small and in one place).
- README states facts that the code contradicts (12-month evaluation, residual definition,
  forecast window). After fixing, regenerate README numbers from the pipeline.

---

## What "way better" looks like

A reviewer reading the improved project should see:

1. **One reproducible pipeline** (`python -m forecasting run --store all`) driven by a config
   file, with seeds, that rebuilds every number and chart from raw data.
2. **Rolling-origin backtesting** of a model ladder: naive → seasonal naive → mean →
   ETS → SARIMA → *correct* SARIMA+LSTM hybrid → global LightGBM with lag/calendar features.
   Metrics: MAE, RMSE, WAPE, MASE, bias, with Diebold–Mariano tests vs the best baseline.
3. **Honest data section**: the synthetic-data diagnostics above, plus results on a real
   retail dataset (M5/Walmart or Corporación Favorita on Kaggle) where models *can* be told apart.
4. **Probabilistic forecasts**: 80/95% intervals, shown as bands on interactive charts.
5. **An actually intelligent dashboard**: per-product accuracy, model chosen, trend vs last year,
   and a reorder suggestion (safety stock = z · σ_forecast · √lead-time).
6. **Engineering quality**: package layout, tests that guard against leakage, CI, Dockerfile,
   env-based config, cached data layer, Jinja template inheritance.
