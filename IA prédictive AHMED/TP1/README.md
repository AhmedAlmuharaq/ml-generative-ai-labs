# TP1 — Data Manipulation with Pandas

## Overview
An introduction to data manipulation and exploratory data analysis (EDA) using the **Iris dataset** and the Pandas library. This lab covers the full data preparation workflow: loading, cleaning, transforming, filtering, and visualizing tabular data.

---

## Objectives
- Load a dataset from scikit-learn and convert it to a Pandas DataFrame.
- Inspect structure, statistics, and data quality.
- Rename, add, and drop columns.
- Filter rows based on conditions.
- Produce histograms, scatter plots, and boxplots with Matplotlib.

---

## File

| File | Description |
|------|-------------|
| `TP1_Pandas_AHMED.ipynb` | Complete notebook: all exercises and outputs |

---

## Dataset

| Property | Value |
|----------|-------|
| Name | Iris (Fisher, 1936) |
| Source | `sklearn.datasets.load_iris` |
| Samples | 150 |
| Features | Sepal length, Sepal width, Petal length, Petal width |
| Target | Species: *setosa*, *versicolor*, *virginica* |

---

## Steps Covered

| # | Task | Key Concept |
|---|------|-------------|
| 1 | Load dataset and display first 5 rows | `pd.DataFrame`, `head()` |
| 2 | Inspect shape and column types | `shape`, `dtypes`, `info()` |
| 3 | Descriptive statistics | `describe()` |
| 4 | Rename columns to cleaner names | `rename()` |
| 5 | Add derived column `PetalRatio` | Feature engineering |
| 6 | Drop unnecessary column `SpeciesId` | `drop()` |
| 7 | Filter rows: `SepalLengthCm >= 5.0` | Boolean indexing |
| 8 | Filter by species (*setosa*) | `value_counts()` |
| 9 | Count samples per species | `groupby()` |
| 10 | Visualizations: histogram, scatter, boxplot | `matplotlib.pyplot` |

---

## Setup

Install dependencies:
```bash
pip install pandas numpy matplotlib scikit-learn
```

No external data files required — the Iris dataset is bundled with scikit-learn.

---

## Key Results

- After filtering `SepalLengthCm >= 5.0`: **128 samples** remain (from 150).
- Species distribution post-filter: *versicolor* 49, *virginica* 49, *setosa* 30.
- `PetalRatio` (length/width) clearly separates *setosa* from other species.

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, manipulation, filtering |
| `matplotlib` | Visualizations (histogram, scatter, boxplot) |
| `scikit-learn` | Iris dataset loader |
