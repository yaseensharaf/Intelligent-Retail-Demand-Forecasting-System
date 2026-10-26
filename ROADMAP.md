# Upgrade Roadmap — copy-paste prompts for Claude Code and Codex

## Setup (once)
1. Copy `AUDIT.md`, `AGENTS.md`, `CLAUDE.md`, `ROADMAP.md` and `scripts/` into the repo root.
2. `git checkout -b phase-0-hygiene` and commit them.
3. Run `python scripts/baseline_check.py` yourself so you've seen the numbers with your own eyes.

**How to split the work between the two tools**
- **Claude Code (VS Code extension)**: the big structural phases (1, 2, 3) — it handles
  multi-file refactors well. Start each phase in plan mode (Shift+Tab) and approve the plan first.
- **Codex**: smaller, well-scoped tasks that can run in parallel (tests, CI, Dockerfile,
  template refactor), and **reviewing Claude Code's PRs** against `AUDIT.md`.
- One branch and one PR per phase. Merge only when `pytest` passes.

---

## Phase 0 — Repository hygiene  *(Codex, ~1 hour)*
> Read AGENTS.md and AUDIT.md §5. Clean up the repository: add `requirements.txt` (pinned:
> pandas, numpy, statsmodels, pmdarima, scikit-learn, tensorflow-cpu or torch, lightgbm,
> flask, pyyaml, pytest, ruff), a Python `.gitignore` that ignores `outputs/`, delete
> `src/tempCodeRunnerFile.py` and the duplicate data files (`H_M copy.csv`, `All_H&M.csv`,
> `M(H&M).csv`), move raw CSVs to `data/raw/` (convert R1/R2 from cp1252 to UTF-8, rename to
> `hm.csv`, `store1.csv`, `store2.csv`), move notebooks to `notebooks/legacy/`. Don't change
> any logic yet. Update paths that break.

## Phase 1 — Reproducible data + evaluation framework  *(Claude Code)*
> Read AGENTS.md and AUDIT.md. Create the `src/forecasting` package per AGENTS.md "Target layout".
> Implement: `data.py` (load each store, validate with explicit checks that log the data-quality
> issues from AUDIT §2, aggregate to a full month-end calendar with zero-fill), `metrics.py`
> (MAE, RMSE, WAPE, MASE vs seasonal-naive, bias, sMAPE, Diebold–Mariano test),
> `models/baselines.py` (naive, seasonal naive, historical mean, drift), and `backtest.py`
> (rolling-origin: origins at end of 2020, 2021, 2022; horizon 12; expanding window).
> Add `config/default.yaml` and a CLI `python -m forecasting backtest --store all` that writes
> `outputs/backtest_<store>.csv` and `reports/results.md`. Add pytest tests: calendar has no gaps,
> no fitted model ever sees data at or after its origin, metrics match hand-computed examples.

## Phase 2 — Models, done correctly  *(Claude Code)*
> Read AGENTS.md "Correct hybrid" and AUDIT §1. Add models behind the common Forecaster interface:
> ETS (statsmodels), SARIMA (pmdarima auto_arima, fit once, no trace), the corrected SARIMA+LSTM
> hybrid (in-sample residuals, recursive residual forecast, blend weight chosen on the last
> validation fold), and a global LightGBM model trained across all products of a store with lag
> 1/2/3/12, rolling mean/std, month-of-year and product features, recursive multi-step. All models
> return mean + 80/95% intervals (SARIMA/ETS native; LightGBM via quantile objective or
> conformal residuals). Seed everything. Run the backtest for all stores and regenerate
> `reports/results.md` with a table per store: every model × every metric, best in bold,
> DM-test p-value vs best baseline. Then `forecast.py`: refit the per-product best model on ALL data
> through Dec 2023 and forecast Jan 2024–Dec 2028 with correctly aligned dates.

## Phase 3 — Real data  *(Claude Code, the biggest boost for the dissertation)*
> Add a loader for a real retail dataset so the model comparison is meaningful. Use the Kaggle
> M5 Forecasting dataset (Walmart): aggregate daily item sales to monthly per item for one store
> (e.g. CA_1) and one department, giving a few hundred series. Download instructions in README
> (Kaggle API), don't commit the data. Run the same backtest and report side by side with the
> synthetic stores. Add an "Data diagnostics" section to `reports/results.md` that prints the
> seasonality R² and consistency checks for every dataset.

## Phase 4 — Dashboard rebuild  *(Codex for templates, Claude Code for the data layer)*
> Read AUDIT §4. Restructure into `src/dashboard`: Flask app factory, `SECRET_KEY` from env,
> debug off by default. `repository.py` loads pipeline outputs (`outputs/forecasts.parquet`,
> `outputs/backtest_*.csv`) once at startup and serves store-aware queries keyed by `product_id`
> slug. Replace the six duplicated templates with `base.html` + small page templates. Remove
> Socket.IO and `/updateAll`. Replace PNG charts with Chart.js (or Plotly) fed by
> `/api/products/<store>/<product_id>/series` returning history, forecast mean and 80/95 bands.
> Pages: Overview (store KPIs: WAPE, best model share, forecast next 12 months vs last 12),
> Product detail (chart with bands, model used, its backtest metrics vs baseline), Trending
> (growth % = next-12 forecast / last-12 actual − 1), Slow movers, and an Inventory page with
> reorder suggestions: safety stock = z · σ · √L, reorder point = mean demand over lead time +
> safety stock, with service level and lead time as form inputs. Keep the light/dark toggle
> (localStorage only). Add tests with Flask's test client for every route.

## Phase 5 — Ship it  *(Codex)*
> Add a Dockerfile (python:3.11-slim, gunicorn), a GitHub Actions workflow running ruff + pytest
> + a smoke backtest on 3 products, and a Makefile (`make data backtest forecast app test`).
> Rewrite README: architecture diagram (mermaid), honest results table pulled from
> `reports/results.md`, data-quality caveats, how to reproduce in three commands, screenshots.

## Phase 6 — Review pass  *(the other tool)*
> Review this branch against AUDIT.md. For every numbered issue, state whether it is fixed,
> partly fixed, or still present, citing file and line. Then look for new leakage, date
> misalignment, or scaling bugs in the models. Don't change code, just report.

---

## What to say in the dissertation / interviews
Turn the findings into the story — it's stronger than the original claim:
"The first version reported 15% MAPE for an ARIMA–LSTM hybrid. Building a proper backtest
showed a flat historical mean scored 12%, the residual pipeline had alignment bugs, and the
dataset had no learnable seasonality. I rebuilt it as a reproducible pipeline with baselines,
rolling-origin evaluation and significance testing, validated it on real Walmart M5 data, and
turned the forecasts into reorder recommendations."
