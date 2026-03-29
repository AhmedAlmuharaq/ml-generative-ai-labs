# TP3 — Classification in Machine Learning

## Overview
A comparative classification lab using the **Breast Cancer Wisconsin dataset**. Three supervised learning models — Decision Tree, Random Forest, and XGBoost — are trained, evaluated, and compared on a binary medical classification task (malignant vs. benign tumour detection).

---

## Objectives
- Load and explore a real medical dataset.
- Perform a stratified train/test split.
- Train three classifiers with appropriate hyperparameters.
- Evaluate models using accuracy, precision, recall, F1-score, and confusion matrices.
- Analyse feature importances to understand model decisions.
- Compare model performance in a summary table.

---

## File

| File | Description |
|------|-------------|
| `Classification en Machine Learning .ipynb` | Complete notebook: training, evaluation, and comparison |

---

## Dataset

| Property | Value |
|----------|-------|
| Name | Breast Cancer Wisconsin (Diagnostic) |
| Source | `sklearn.datasets.load_breast_cancer` |
| Samples | 569 |
| Features | 30 numeric cell-nucleus measurements |
| Classes | Malignant (212) / Benign (357) |
| Split | 80% train / 20% test (stratified) |

---

## Models Compared

| Model | Key Hyperparameters |
|-------|---------------------|
| Decision Tree | `max_depth=4`, `min_samples_leaf=5` |
| Random Forest | `n_estimators=300`, `min_samples_leaf=2` |
| XGBoost | `n_estimators=400`, `lr=0.05`, `max_depth=4`, `subsample=0.9` |

> XGBoost requires a separate installation (see Setup below).

---

## Steps Covered

| # | Task | Key Concept |
|---|------|-------------|
| 1 | Load dataset and explore class distribution | `load_breast_cancer` |
| 2 | Stratified train/test split | `train_test_split(stratify=y)` |
| — | Define shared evaluation helper | Confusion matrix + 4 metrics |
| 3 | Train and evaluate Decision Tree | `DecisionTreeClassifier` |
| 4 | Train and evaluate Random Forest | `RandomForestClassifier` |
| 5 | Train and evaluate XGBoost (if installed) | `XGBClassifier` |
| 6 | Build comparison table sorted by F1 | `pd.DataFrame` |
| 7 | Plot top-12 feature importances per model | `feature_importances_` |
| 8 | Short written conclusion | Best model identification |

---

## Evaluation Metrics

All metrics are computed with **malignant (class 0) as the positive class**, which is the clinically relevant choice (false negatives are more dangerous than false positives).

| Metric | Best result (this run) |
|--------|----------------------|
| Accuracy | Random Forest: **95.6%** |
| Precision (malignant) | Random Forest: **95.1%** |
| Recall (malignant) | Random Forest: **92.9%** |
| F1-score (malignant) | Random Forest: **93.9%** |

---

## Setup

```bash
pip install scikit-learn pandas numpy matplotlib
pip install xgboost   # optional — required for Step 5
```

No external data files required.

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `scikit-learn` | Dataset, classifiers, metrics, confusion matrix |
| `xgboost` | Gradient boosting classifier (optional) |
| `pandas` | Results summary table |
| `numpy` | Numerical operations |
| `matplotlib` | Confusion matrices, feature importance plots |
