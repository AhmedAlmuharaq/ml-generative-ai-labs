# TP6 — Time Series Analysis and Forecasting

## Overview
A four-part time series lab using the **Bike Sharing Demand dataset** (Capital Bikeshare, Washington D.C., 2011–2012). Starting from raw hourly rental data, the lab progressively builds from exploratory visualisation through classical statistical decomposition and ARIMA modelling to supervised machine learning forecasting — providing a complete view of the time series analysis pipeline.

---

## Objectives
- Load, clean, and align an irregular hourly time series.
- Visualise temporal patterns: hourly, daily, weekly, and yearly.
- Decompose a series into trend, seasonality, and residuals.
- Test for stationarity and apply differencing.
- Build and evaluate AR / ARIMA models.
- Compare statistical and supervised ML forecasting approaches.

---

## Files

| File | Part | Description |
|------|------|-------------|
| `TP6.1 Demand exploration and visualization.ipynb` | 1 | Data loading, cleaning, and exploratory visualisation |
| `TP6_2_decomposition_stationarity.ipynb` | 2 | Trend/seasonality decomposition and stationarity tests |
| `TP6_3_AR_ARIMA.ipynb` | 3 | AR and ARIMA model fitting and evaluation |
| `TP6_4_Supervised_Comparison.ipynb` | 4 | Supervised ML models vs ARIMA — performance comparison |

---

## Dataset

| Property | Value |
|----------|-------|
| Name | Bike Sharing Demand |
| File | `train.csv` (place in the project root or update `ROOT` in the notebook) |
| Time span | 2011-01-01 → 2012-12-19 |
| Raw observations | 10,886 hourly records |
| After `asfreq("h")` | 17,256 hourly slots (gaps filled with `ffill`) |
| Target variable | `count` — total hourly bike rentals |

> **Important:** place `train.csv` somewhere under the `ROOT` path defined in `TP6.1`. The notebook searches for it automatically.

---

## Part 1 — Exploration and Visualisation

| Task | Key Concept |
|------|-------------|
| Load CSV with `parse_dates` | `DatetimeIndex` |
| Enforce hourly frequency | `asfreq("h")` |
| Fill gaps via forward-fill | `ffill()` |
| Distribution: histogram, boxplot | Right-skewed demand |
| Full time series plot (2011–2012) | Overall trend |
| Year-by-year comparison | Growth between years |
| Hourly profile (mean by hour) | Two daily peaks: 8h and 17–18h |
| Day-of-week profile | Weekday vs weekend patterns |
| Lag plots (lag=1, lag=24) | Autocorrelation structure |
| ACF / PACF (48 lags) | Dominant lags at 1, 24, 168 |

---

## Part 2 — Decomposition and Stationarity

| Task | Key Concept |
|------|-------------|
| Classical decomposition | Trend + seasonality + residual |
| STL decomposition | Robust seasonal-trend decomposition |
| ADF test | Augmented Dickey-Fuller stationarity test |
| First differencing | Remove trend to achieve stationarity |
| Seasonal differencing (lag=24) | Remove daily seasonality |

---

## Part 3 — AR / ARIMA Modelling

| Task | Key Concept |
|------|-------------|
| AR model | Autoregressive model (p lags) |
| ARIMA(p, d, q) | Integrated ARMA with differencing order d |
| Model selection | AIC / BIC criteria |
| Residual diagnostics | Ljung-Box test, residual ACF |
| Forecasting | Rolling one-step ahead predictions |
| Evaluation | MAE, RMSE on test set |

---

## Part 4 — Supervised ML Comparison

| Task | Key Concept |
|------|-------------|
| Feature engineering | Lag features, rolling statistics, calendar features |
| Models compared | Random Forest, Gradient Boosting, XGBoost |
| Train/test split | Time-based (no shuffle) |
| Metrics | MAE, RMSE, R² |
| Comparison table | Statistical vs ML forecasting |

---

## Setup

```bash
pip install pandas numpy matplotlib scikit-learn statsmodels xgboost
```

Ensure `train.csv` is available and that the `ROOT` path variable in `TP6.1` points to its parent directory.

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Time series loading, resampling, `DatetimeIndex` |
| `numpy` | Numerical operations |
| `matplotlib` | All visualisations |
| `statsmodels` | ACF, PACF, decomposition, ADF test, ARIMA |
| `scikit-learn` | Supervised regressors, metrics, cross-validation |
| `xgboost` | Gradient boosting regressor (Part 4) |
