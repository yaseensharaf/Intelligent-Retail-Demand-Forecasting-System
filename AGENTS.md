# AGENTS.md — instructions for coding agents (Codex, Claude Code)

## Project
Final-year dissertation project (UWE Bristol, UXCFXK-30-3): monthly product-level retail
demand forecasting for three stores + a Flask dashboard. Being upgraded from notebooks to a
reproducible, well-evaluated system. **Read `AUDIT.md` before changing anything** — it lists
every known bug with evidence. Do not reintroduce any issue listed there.

## Ground rules
- Python 3.11. Use pandas ≥ 2.2 idioms (`freq="ME"`, not `"M"`).
- Never edit raw files in `data/raw/`. All derived data is produced by the pipeline into
  `outputs/` (git-ignored) and is reproducible from a single command.
- No train/test leakage: every model is fit only on data strictly before the forecast origin.
  Tests must enforce this.
- Every model is compared against naive, seasonal-naive and historical-mean baselines.
  Never report a result without the baselines next to it.
- Set seeds (`numpy`, `random`, `tensorflow`/`torch`) from config.
- Keep forecasts as floats until the final display layer. Don't round or floor before scoring.
- Read CSVs with an explicit encoding. The raw files in `data/raw/` are UTF-8. The old generated file
  `data/ALLRetail_Store_2.csv` is cp1252; it is replaced by pipeline outputs, so never read it as UTF-8.
- Product names are matched exactly; normalise to a `product_id` slug for URLs and file names.
  Never lowercase display names.
- Small, reviewable commits. One phase per branch (`phase-1-pipeline`, …). Run `pytest` and
  `ruff check .` before every commit.
- If a change alters reported numbers, update `reports/results.md` from pipeline output; never
  hand-type metrics into the README.

## Target layout
```
pyproject.toml / requirements.txt
config/default.yaml                 # stores, paths, horizon, backtest origins, seeds, model params
src/forecasting/
    data.py        # load, validate (pandera or explicit checks), clean, monthly aggregation with full calendar
    features.py    # lags, rolling stats, calendar features for global models
    models/        # baselines.py, ets.py, sarima.py, hybrid.py, gbm.py — common fit/predict interface
    backtest.py    # rolling-origin evaluation
    metrics.py     # MAE, RMSE, WAPE, MASE, bias, sMAPE; Diebold–Mariano
    forecast.py    # refit on all data, horizon forecasts + prediction intervals
    cli.py         # `python -m forecasting run|backtest|report`
src/dashboard/
    app.py         # app factory, config from env
    repository.py  # loads pipeline outputs once, cached; store-aware queries
    templates/base.html + page templates (Jinja inheritance)
    static/
tests/
scripts/baseline_check.py
notebooks/          # exploratory only; import from src/forecasting, no pipeline logic
reports/results.md  # generated
```

## Model interface
```python
class Forecaster(Protocol):
    name: str
    def fit(self, y: pd.Series) -> "Forecaster": ...          # monthly, freq="ME", no gaps
    def predict(self, h: int) -> pd.DataFrame: ...           # columns: mean, lo80, hi80, lo95, hi95
```

## Correct hybrid (replacement for the notebook version)
1. Fit SARIMA on train. Residuals = `fit.resid` (in-sample), drop the first `s+d` burn-in values.
2. Train the residual model (LSTM or GBM) on sliding windows of those residuals, with its own
   scaler fit on train residuals only.
3. Forecast residuals **recursively** h steps ahead, starting from the last `window` train residuals.
4. Final = SARIMA forecast + residual forecast, aligned by date. Choose the blend weight (if any)
   on a validation fold, not by hand.

## Commands
```
pip install -r requirements.txt
python -m forecasting backtest --store all
python -m forecasting run --store all
python -m forecasting report
pytest -q
flask --app src/dashboard/app run
```

## Definition of done for any task
Tests pass, lint passes, the pipeline still runs end-to-end on all stores, and
`reports/results.md` is regenerated if numbers changed.
