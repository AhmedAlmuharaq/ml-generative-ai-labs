# TP4 — XGBoost Advanced

## Overview
An advanced lab dedicated to fine-tuning and deeply understanding **XGBoost** for binary classification. Using the Breast Cancer Wisconsin dataset, this lab explores the impact of the learning rate, early stopping, and validation monitoring on model performance and generalisation.

---

## Objectives
- Understand the role of `learning_rate` and `n_estimators` in gradient boosting.
- Use a **train/validation/test split** to monitor overfitting.
- Apply **early stopping** to automatically find the optimal number of trees.
- Compare multiple learning rate configurations and select the best.
- Read and interpret XGBoost evaluation curves (logloss over boosting rounds).

---

## File

| File | Description |
|------|-------------|
| `TP4 XGBoost Advanced.ipynb` | Complete notebook: configuration sweep, early stopping, learning curves |

---

## Dataset

| Property | Value |
|----------|-------|
| Name | Breast Cancer Wisconsin (Diagnostic) |
| Source | `sklearn.datasets.load_breast_cancer` |
| Samples | 569 |
| Features | 30 numeric measurements |
| Split | 64% train / 16% validation / 20% test |

---

## Steps Covered

| # | Task | Key Concept |
|---|------|-------------|
| 1 | Three-way data split (train / val / test) | `train_test_split` × 2 |
| 2 | Define `run_config()` helper | Flexible XGBoost training with early stopping |
| 3 | Sweep learning rates: 0.03, 0.10, 0.20 | Hyperparameter sensitivity |
| 4 | Compare configs: best trees, accuracy, AUC, logloss | `pd.DataFrame` summary |
| 5 | Plot validation logloss curves per config | `evals_result()` |
| 6 | Train final model on full train set (no val) | Generalisation comparison |

---

## Key Concepts

| Concept | Explanation |
|---------|-------------|
| `learning_rate` | Step size for each boosting round. Lower = more trees needed but better generalisation. |
| `early_stopping_rounds` | Stop training if validation metric does not improve for N consecutive rounds. |
| `eval_set` | Pass `[(X_train, y_train), (X_val, y_val)]` to monitor both sets per round. |
| `evals_result()` | Returns the full history of train/val metrics — used to plot learning curves. |
| AUC | Area Under the ROC Curve — robust metric for imbalanced binary classification. |
| Logloss | Log-loss on probabilities — lower is better, used as the stopping criterion. |

---

## Results Summary

| Learning Rate | Best Trees | Accuracy | AUC |
|---------------|------------|----------|-----|
| 0.03 | (high) | high | high |
| 0.10 | medium | ~96% | ~99% |
| 0.20 | (low) | competitive | high |

> Exact values depend on early stopping — run the notebook to see the full table.

---

## Setup

```bash
pip install xgboost scikit-learn pandas numpy matplotlib
```

No external data files required.

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `xgboost` | `XGBClassifier`, `EarlyStopping` callback, `evals_result()` |
| `scikit-learn` | Data loading, train/test split, accuracy, AUC, log-loss |
| `pandas` | Configuration comparison table |
| `numpy` | Array operations |
| `matplotlib` | Validation logloss learning curves |
