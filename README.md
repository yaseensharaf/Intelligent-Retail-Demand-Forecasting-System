<div align="center">

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
<img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white"/>
<img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>

# Intelligent Retail Demand Forecasting System

**A hybrid ARIMA–LSTM forecasting pipeline and Flask dashboard for product-level retail sales**

</div>

---

## Table of Contents

1. [Overview](#overview)
2. [Results](#results)
3. [Dashboard](#dashboard)
4. [How the Forecasting Works](#how-the-forecasting-works)
5. [Project Structure](#project-structure)
6. [Getting Started](#getting-started)
7. [Data](#data)
8. [API Endpoints](#api-endpoints)
9. [Known Limitations & Future Work](#known-limitations--future-work)
10. [Author & License](#author--license)

---

## Overview

Retailers that forecast poorly either overstock (tying up capital) or run out of stock (losing sales). This project forecasts **monthly unit sales per product** and puts the results in a web dashboard so a manager can see what is trending, what is underperforming, and what other stores are selling well.

- **Forecasting:** each product gets a seasonal **ARIMA** model, blended with an **LSTM** trained on the residual series.
- **Scope:** **174 products** across three stores: **H&M (80)**, **Retail Store 1 (54)**, and **Retail Store 2 (40)**.
- **Data:** monthly sales history from **2015 to 2023**, forecast forward to **September 2028** (57 months).
- **Dashboard:** a Flask app that lists products, shows each product's forecast chart, and ranks trending, lowest-selling, and top-selling items.

---

## Results

Accuracy was measured **for H&M only**, on a held-out year: models were trained on 2015–2022 and scored against the 12 months of 2023, per product.

| Metric (80 H&M products) | Mean | Median | Min – Max |
|---|---|---|---|
| **MAPE** | **15.2%** | 15.6% | 7.2% – 25.7% |
| MAE (units / month) | 129.9 | 129.1 | 57.6 – 342.3 |
| RMSE (units / month) | 155.7 | 152.8 | 69.5 – 379.5 |

- Every product has a MAPE under **30%**, and **90%** of products are under **20%**.
- Best: Hat (7.2%), Lace-Up Boots (7.9%), Sandals (8.6%). Hardest: Wool-Cotton Jacket (25.7%), Wool Trousers (24.6%), Fluffy Jacket (24.3%).
- Per-product figures are in [`Sales_Forecast_Results_with_Accuracy/H&M_accuracy_metrics.csv`](Sales_Forecast_Results_with_Accuracy/H&M_accuracy_metrics.csv).
- Retail Store 1 and Retail Store 2 models are trained on all data through 2023 (no held-out year), so no accuracy metrics are reported for them.

---

## Dashboard

A Flask + Flask-SocketIO web app (`src/app.py`) with server-rendered Jinja2 pages, a custom-CSS responsive layout, and a light/dark theme toggle that is remembered in the browser.

| Page | Route | What it shows |
|---|---|---|
| **Home** | `/` | H&M product cards with **search** (name or category) and a store filter. **Show Image** displays that product's forecast chart. |
| **All Products** | `/all_products.html` | Every product from all three stores (174), with search and store filter. |
| **Trending** | `/trending.html` | Best sellers across all stores since Jan 2023: the top 5 per store, ranked to a top 15 overall. |
| **Top-Selling (Recommend)** | `/recommend.html` | The top 5 products from each of Retail Store 1 and Retail Store 2 since Jan 2023. |
| **Lowest-Selling** | `/lowest.html` | The 5 lowest-selling H&M products since Jan 2023. |
| **Help** | `/help.html` | A plain-language guide to using each page. |

> "Since Jan 2023" means 2023 actual sales plus the 2024–2028 forecast, summed per product.

Forecast charts show historical sales (blue) and predicted sales (red, dashed) and are pre-rendered by the notebooks as PNG images.

---

## How the Forecasting Works

Each product's sales are aggregated to **monthly units sold** and modeled separately. Products with fewer than **24 months** of data are skipped.

1. **ARIMA.** `pmdarima.auto_arima` fits a seasonal model (`m=12`, `d=1`, `D=1`, stepwise search, AIC criterion) per product.
2. **Residuals.** The gap between observed sales and the ARIMA output is standardized with `StandardScaler`.
3. **LSTM.** A network trained on the residual series using a **3-month sliding window**:
   ```
   Input (3 × 1) → LSTM(100, ReLU) → Dropout(0.2) → LSTM(50, ReLU) → Dropout(0.2) → Dense(1)
   ```
   Adam optimizer, MSE loss, 100 epochs, batch size 16.
4. **Blend.** The final forecast is a weighted sum, rounded to whole units and floored at 1:
   ```
   forecast = w_arima × ARIMA + w_lstm × LSTM
   ```

| Notebook | Store | Products | Blend (ARIMA / LSTM) | Training data | Held-out metrics |
|---|---|---|---|---|---|
| `H_M.ipynb` | H&M | 80 | 90% / 10% | 2015–2022 (2023 held out) | Yes |
| `R1.ipynb` | Retail Store 1 | 54 | 90% / 10% | 2015–2023 | No |
| `R2.ipynb` | Retail Store 2 | 40 | 95% / 5% | 2015–2023 | No |

The sliding window consumes 3 of the 60 requested steps, so each forecast file contains **57 months (Jan 2024 – Sep 2028)**.

Each notebook then merges actual and forecast sales per product and combines them into one file per store (`HM_All_Product_Sales`, `ALLRetail_Store_1`, `ALLRetail_Store_2`), which the dashboard reads from `data/`.

---

## Project Structure

```
Intelligent-Retail-Demand-Forecasting-System/
│
├── src/
│   ├── app.py                       # Flask + Flask-SocketIO application (entry point)
│   ├── templates/                   # Jinja2 pages: index, all_products, trending,
│   │                                #   recommend, lowest, help
│   ├── static/
│   │   ├── css/Main.css             # Styles (light / dark theme)
│   │   ├── js/main.js               # Theme toggle, search/filter, image viewer
│   │   └── images/                  # Forecast chart PNGs used by the pages
│   └── tempCodeRunnerFile.py        # Editor scratch file (not used)
│
├── static/images/                   # Second copy of the chart PNGs; /product_image
│                                    #   checks this path relative to the working directory
│
├── data/
│   ├── H_M.csv                      # Raw H&M transactions
│   ├── R1(Main).csv                 # Raw Retail Store 1 transactions
│   ├── R2.csv                       # Raw Retail Store 2 transactions
│   ├── HM_All_Product_Sales.csv     # Monthly history + forecast, H&M   (used by dashboard)
│   ├── ALLRetail_Store_1.csv        # Monthly history + forecast, Store 1 (used by dashboard)
│   ├── ALLRetail_Store_2.csv        # Monthly history + forecast, Store 2 (used by dashboard)
│   └── All_H&M.csv, M(H&M).csv,     # Alternate copies / subsets (not used by the app)
│       H_M copy.csv
│
├── Sales_Forecast_Results_with_Accuracy/   # H&M: 80 forecast CSVs + PNGs + accuracy metrics
├── Retail_Store_1/                         # Store 1: 54 forecast CSVs + PNGs
├── Retail_Store_2/                         # Store 2: 40 forecast CSVs + PNGs
│
├── H_M.ipynb                        # H&M pipeline (train/test split, metrics, forecast, merge)
├── R1.ipynb                         # Retail Store 1 pipeline
├── R2.ipynb                         # Retail Store 2 pipeline
└── README.md
```

---

## Getting Started

Developed with Python 3.11.

### Run the dashboard

```bash
# 1. Clone
git clone https://github.com/yaseensharaf/Intelligent-Retail-Demand-Forecasting-System.git
cd Intelligent-Retail-Demand-Forecasting-System

# 2. (Recommended) virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dashboard dependencies
pip install flask flask-socketio pandas

# 4. Start the app -- run this from the repository root
python src/app.py
```

Open **http://127.0.0.1:5001**.

> **Run from the repository root.** The app reads `data/` and `static/images/` using paths relative to the working directory, so starting it from inside `src/` will show empty pages.

### Regenerate the forecasts (optional)

The forecast outputs are already included, so this is only needed to retrain.

```bash
pip install pandas numpy matplotlib statsmodels pmdarima scikit-learn tensorflow openpyxl jupyter
jupyter notebook
```

1. The notebooks read `H_M.csv`, `R1(Main).csv`, and `R2.csv` from their own working directory. Copy them from `data/` next to the notebooks (or edit the `file_path` line).
2. Run `H_M.ipynb`, `R1.ipynb`, and `R2.ipynb`. Each writes per-product forecast CSVs and PNGs, then merges and combines them into one Excel file per store.
3. Save the combined series as CSV in `data/` (`HM_All_Product_Sales.csv`, `ALLRetail_Store_1.csv`, `ALLRetail_Store_2.csv`) so the dashboard picks them up.

Training runs an ARIMA search plus a 100-epoch LSTM for every product, so a full run takes a while.

---

## Data

| File | Rows | Description |
|---|---|---|
| `data/H_M.csv` | 674,517 | H&M transactions: `TransactionID`, `ProductID`, `ProductName`, `Category`, `Price`, `QuantitySold`, `SaleDate`, `Season`, `Year` |
| `data/R1(Main).csv` | 70,000 | Store 1 transactions (same columns plus `DiscountApplied`, `TransactionAmount`, `CustomerLoyalty`) |
| `data/R2.csv` | 50,000 | Store 2 transactions (same columns as Store 1) |
| `data/HM_All_Product_Sales.csv` | 13,200 | H&M monthly series, 80 products: `ProductName`, `Date`, `Sales` (history + forecast) |
| `data/ALLRetail_Store_1.csv` | 8,908 | Store 1 monthly series, 54 products |
| `data/ALLRetail_Store_2.csv` | 4,744 | Store 2 monthly series, 40 products |
| `Sales_Forecast_Results_with_Accuracy/*_forecast_2024_2028.csv` | 57 each | `Date`, `Predicted Sales` |

`Sales` is **units sold per month**. Forecast files use month-end dates.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/product_image/<product_name>` | Returns `{"image_url": ...}` for the product's chart image (spaces map to underscores), or a 404 error. |
| GET | `/product_sales/<product_name>` | Returns `{"dates": [...], "quantities": [...]}` with the combined historical + forecast series across stores, or a 404 error. |
| POST | `/updateAll` | Saves the posted JSON to `update_data.json` and broadcasts a Socket.IO `data_updated` event. |
| POST | `/toggle-dark-mode` | Flips a dark-mode flag in the server session. |

---

## Known Limitations & Future Work

**Limitations**

- **Accuracy is measured for H&M only.** Stores 1 and 2 have no held-out evaluation, and no ARIMA-only baseline is included, so the gain from the hybrid step is not quantified.
- **The LSTM sees very little data.** For H&M it trains on 9 windows built from a 12-month residual series, which is why its weight in the blend is small (10%, or 5% for Store 2).
- **H&M final forecasts are not refit on 2023.** The H&M model is trained on 2015–2022 and reused for the 2024–2028 forecast, whereas Stores 1 and 2 are trained through 2023.
- **Run from the repository root** (see Getting Started). Data paths are relative.
- **Case-sensitive file systems.** The app looks for `data/R1(main).csv`, but the file is named `R1(Main).csv`. On Linux, Retail Store 1 products therefore appear as "Uncategorized"; renaming the file fixes it.
- **Local development settings.** `app.py` runs with `debug=True` and a hard-coded session secret, so it is not production-ready as is.
- **No automated tests** are included in the repository.

**Future work**

- Add an ARIMA-only baseline and evaluate every store on a held-out year.
- Refit on all available data before producing the final forecast.
- Add a `requirements.txt`, unit tests (pytest), a Dockerfile, and a CI workflow.
- Move the CSV storage to a database (for example PostgreSQL) and load the secret key from the environment.
- Try additional models (Prophet, XGBoost, Temporal Fusion Transformer) on volatile products.

---

## Author & License

**Yaseen Sharaf** · [GitHub](https://github.com/yaseensharaf) · [LinkedIn](https://www.linkedin.com/in/yaseensharaf04/)

```
Copyright © 2025 Yaseen Sharaf. All rights reserved.
Developed as a final-year academic project — UXCFXK-30-3 Digital Systems Project.
Unauthorised commercial use or redistribution is not permitted.
```
