# TP5 — Regression, Cross-Validation, and Complexity Control

## Overview
A comprehensive regression lab using the **California Housing dataset**. This lab covers the full supervised regression workflow: baseline evaluation, k-fold cross-validation for model selection, and systematic complexity control through the `max_depth` hyperparameter — with a synthetic experiment to visualise underfitting and overfitting.

---

## Objectives
- Distinguish regression from classification tasks and choose appropriate metrics.
- Train and evaluate a baseline Decision Tree regressor (MAE, RMSE, R²).
- Apply **k-fold cross-validation** for robust model comparison.
- Compare Decision Tree, Random Forest, and Gradient Boosting via CV.
- Sweep `max_depth` from 2 to 10 and identify the bias–variance trade-off.
- Visualise overfitting and underfitting on a synthetic sinusoidal dataset.

---

## File

| File | Description |
|------|-------------|
| `Regression, cross-validation, and complexity control.ipynb` | Complete notebook: all 4 parts with code and analysis |

---

## Dataset

| Property | Value |
|----------|-------|
| Name | California Housing |
| Source | `sklearn.datasets.fetch_california_housing` |
| Samples | 20,640 |
| Features | 8 (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude) |
| Target | Median house value (in $100k units) |
| Split | 80% train / 20% test |

---

## Structure

### Part 1 — Baseline Regression
| Task | Details |
|------|---------|
| Train/test split | `test_size=0.2`, `random_state=42` |
| Baseline model | `DecisionTreeRegressor` (default, no depth limit) |
| Metrics | MAE, RMSE, R² |

### Part 2 — Cross-Validation
| Task | Details |
|------|---------|
| K-Fold setup | `n_splits=5`, `shuffle=True`, `random_state=42` |
| Scoring | `neg_root_mean_squared_error` |
| Model comparison | Decision Tree vs Random Forest vs Gradient Boosting |

### Part 3 — Complexity Control (`max_depth`)
| Task | Details |
|------|---------|
| Depth sweep | `max_depth` from 2 to 10 on Decision Tree |
| Visualisation | CV RMSE curve vs depth (bias–variance plot) |
| Synthetic experiment | `y = sin(x) + noise` — depths 2, 4, 6, 10 |

### Part 4 — Synthesis
Written analysis covering regression vs classification, the role of CV, and the strategy for choosing `max_depth`.

---

## Key Metrics Reference

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MAE | Mean absolute error | Average error magnitude (same unit as target) |
| RMSE | √(MSE) | Penalises large errors more than MAE |
| R² | 1 − SS_res/SS_tot | Fraction of variance explained (1 = perfect, 0 = baseline mean) |

---

## Setup

```bash
pip install scikit-learn pandas numpy matplotlib
```

No external data files required.

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `scikit-learn` | Dataset, regressors, cross-validation, metrics |
| `pandas` | Results comparison tables |
| `numpy` | Numerical operations, synthetic data generation |
| `matplotlib` | Learning curves, depth sweep plots, overfitting visualisation |
