# TP2 — K-Means Clustering

## Overview
An end-to-end unsupervised learning lab applying **K-Means clustering** to the Mall Customers dataset. The goal is to segment customers into meaningful behavioural groups based on their annual income and spending score, without using any pre-existing labels.

---

## Objectives
- Load and explore a real-world customer dataset.
- Apply the **Elbow Method** to determine the optimal number of clusters.
- Train a K-Means model and assign cluster labels.
- Analyse and interpret each customer segment.
- Visualize clusters and centroids on a 2D scatter plot.

---

## File

| File | Description |
|------|-------------|
| `TP2 – Clustering par K-Means .ipynb` | Complete notebook: all steps and outputs |

---

## Dataset

| Property | Value |
|----------|-------|
| Name | Mall Customers |
| Source | [satishgunjal/datasets on GitHub](https://raw.githubusercontent.com/satishgunjal/datasets/master/Mall_Customers.csv) |
| Samples | 200 customers |
| Features used | `AnnualIncome (k$)`, `SpendingScore (1–100)` |
| Missing values | None |

---

## Steps Covered

| # | Task | Key Concept |
|---|------|-------------|
| 1 | Import libraries | `pandas`, `numpy`, `matplotlib`, `KMeans` |
| 2 | Load dataset from URL, check shape | `pd.read_csv` |
| 3 | Descriptive statistics and null check | `describe()`, `isnull()` |
| 4 | Rename columns for clarity | `rename()` |
| 5 | Scatter plot of `AnnualIncome` vs `SpendingScore` | Data exploration |
| 6 | Build feature matrix `X` | `df.loc[:, [...]]` |
| 7 | Elbow Method — inertia for k = 1 to 10 | `KMeans.inertia_` |
| 8 | Train K-Means with optimal k = 5 | `fit_predict()` |
| 9 | Segment analysis: counts, mean income, mean score | `groupby().agg()` |
| 10 | Cluster visualization with centroids | `plt.scatter` |

---

## Setup

```bash
pip install pandas numpy matplotlib scikit-learn
```

The dataset is fetched directly from a public URL — no local file required.

---

## Key Results

| Cluster | Profile | Mean Income | Mean Score |
|---------|---------|-------------|------------|
| 0 | Low income, low spenders | ~26 k$ | ~21 |
| 1 | Low income, high spenders | ~26 k$ | ~79 |
| 2 | High income, low spenders | ~88 k$ | ~17 |
| 3 | High income, high spenders | ~86 k$ | ~82 |
| 4 | Mid income, mid spenders | ~55 k$ | ~50 |

The Elbow Method suggests **k = 5** as the optimal number of clusters.

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Scatter plots, elbow curve |
| `scikit-learn` | `KMeans` clustering algorithm |
