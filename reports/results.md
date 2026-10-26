# Results

## Data diagnostics

| store | rows | products | year_mismatch | season_mismatch | duplicate_transaction_ids | amount_consistency | multi_category_products | seasonality_r2_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hm | 674517 | 80 | 0.0 | 0.0 | 0 |  | 2 | 0.133 |
| store1 | 70000 | 54 | 0.0 | 0.749 | 21575 | 0.033 | 0 | 0.084 |
| store2 | 50000 | 40 | 0.886 | 0.749 | 11836 | 0.528 | 0 | 0.17 |

## H&M

### Metrics (mean over products & origins, best in bold)

| model | mae_mean | rmse_mean | wape_mean | bias_mean | smape_mean | mape_mean | mase_mean | coverage80_mean | coverage95_mean | wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drift | 135.563 | 159.399 | 16.764 | 18.947 | 16.636 | 17.343 | 1.058 | 0.973 | 0.998 | 24 |
| historical_mean | **92.657** | **113.296** | **11.423** | -1.158 | **11.496** | **11.820** | **0.722** | 0.795 | 0.95 | 174 |
| naive | 130.163 | 153.829 | 16.091 | 17.951 | 15.99 | 16.667 | 1.016 | 0.973 | 0.998 | 26 |
| seasonal_naive | 127.884 | 157.205 | 15.769 | -1.092 | 15.873 | 16.196 | 0.996 | 0.797 | 0.95 | 16 |

### Win counts and Diebold-Mariano vs historical_mean

| model | reference | dm_stat | p_value |
| --- | --- | --- | --- |
| naive | historical_mean | 12.1734 | 0.0 |
| seasonal_naive | historical_mean | 20.3567 | 0.0 |
| drift | historical_mean | 12.2203 | 0.0 |

## Retail Store 1

### Metrics (mean over products & origins, best in bold)

| model | mae_mean | rmse_mean | wape_mean | bias_mean | smape_mean | mape_mean | mase_mean | coverage80_mean | coverage95_mean | wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drift | 45.924 | 54.237 | 41.275 | -5.786 | 44.203 | 47.5 | 1.058 | 0.975 | 0.994 | 12 |
| historical_mean | **31.257** | **38.504** | **27.963** | 0.234 | **28.791** | **36.168** | **0.720** | 0.8 | 0.945 | 121 |
| naive | 43.891 | 52.208 | 39.449 | -5.248 | 41.948 | 45.553 | 1.012 | 0.973 | 0.994 | 19 |
| seasonal_naive | 43.696 | 53.921 | 38.823 | 0.554 | 39.519 | 46.699 | 1.003 | 0.806 | 0.949 | 10 |

### Win counts and Diebold-Mariano vs historical_mean

| model | reference | dm_stat | p_value |
| --- | --- | --- | --- |
| naive | historical_mean | 8.0294 | 0.0 |
| seasonal_naive | historical_mean | 13.4207 | 0.0 |
| drift | historical_mean | 8.24 | 0.0 |

## Retail Store 2

### Metrics (mean over products & origins, best in bold)

| model | mae_mean | rmse_mean | wape_mean | bias_mean | smape_mean | mape_mean | mase_mean | coverage80_mean | coverage95_mean | wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drift | 35.611 | 42.392 | 101.622 | -5.936 | 41.254 | 34.939 | 1.494 | 0.966 | 0.985 | 44 |
| historical_mean | **25.451** | **31.528** | **67.155** | -3.009 | 101.631 | **25.590** | **1.270** | 0.871 | 0.96 | 53 |
| naive | 33.948 | 40.728 | 96.313 | -4.758 | 30.38 | 33.46 | 1.437 | 0.967 | 0.985 | 11 |
| seasonal_naive | 35.592 | 43.925 | 85.768 | -1.851 | **27.426** | 36.149 | 1.391 | 0.862 | 0.945 | 12 |

### Win counts and Diebold-Mariano vs historical_mean

| model | reference | dm_stat | p_value |
| --- | --- | --- | --- |
| naive | historical_mean | 4.5748 | 0.0 |
| seasonal_naive | historical_mean | 6.9631 | 0.0 |
| drift | historical_mean | 4.8499 | 0.0 |

## Reference: legacy hybrid

The notebook-era ARIMA-LSTM hybrid reported **15.2% MAPE** (scored on Jan-Sep 2023 only (9 of 12 months), H&M; see AUDIT.md). It is not reproduced by this pipeline.
