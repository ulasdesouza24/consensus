# ConsensusFS

> **Don't trust a single feature selection method — run them all, pick the best.**

ConsensusFS is an **ensemble / consensus feature selection library** for machine learning projects. It runs SHAP, LOFO, Permutation Importance, and Correlation **in parallel**, then intelligently combines the results to produce a reliable **Meta-Importance score**.

Fully Scikit-Learn compatible: works with `fit`, `transform`, `fit_transform`, and Pipelines out of the box.

---

## 📋 Table of Contents

- [Why ConsensusFS?](#why-consensusfs)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [API Reference](#api-reference)
- [Supported Methods](#supported-methods)
- [Aggregation Strategies](#aggregation-strategies)
- [Advanced Usage](#advanced-usage)
- [Sklearn Pipeline Integration](#sklearn-pipeline-integration)
- [Visualization](#visualization)
- [Dependencies](#dependencies)
- [FAQ](#faq)

---

## Why ConsensusFS?

Every feature selection method measures something different — and has different blind spots:

| Method | What It Measures | Weakness |
|--------|-----------------|----------|
| Correlation | Linear relationship with target | Misses non-linear relationships |
| Permutation | Impact on model performance | Can be misleading on overfit models |
| SHAP | Each feature's contribution to predictions | Model-dependent |
| LOFO | Effect on CV score when feature is removed | Slow; noisy on small datasets |

**ConsensusFS** combines these methods to:
- Never fall into any single method's blind spot
- Select features that multiple methods agree are important
- Produce more **stable** and **reliable** results than any individual method

---

## Installation

```bash
pip install consensus-fs
```

**To install dependencies separately:**

```bash
pip install scikit-learn>=1.0.0 pandas>=1.0.0 numpy>=1.18.0 shap>=0.40.0 lofo-importance>=0.3.0 joblib>=1.0.0 seaborn>=0.11.0 matplotlib>=3.3.0
```

> **Python requirement:** >= 3.8

---

## Quick Start

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from consensusfs import ConsensusSelector

# Prepare data
X, y = make_classification(n_samples=500, n_features=20, n_informative=5, random_state=42)
X_df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(20)])

# Define model and selector
model = RandomForestClassifier(n_estimators=100, random_state=42)

selector = ConsensusSelector(
    estimator=model,
    methods=['correlation', 'permutation', 'shap'],  # methods to use
    n_features_to_select=10,                          # how many features to keep
    n_jobs=-1                                         # use all CPU cores
)

# Fit and transform
X_selected = selector.fit_transform(X_df, y)

print("Selected features:", selector.best_features_)
print("New shape:", X_selected.shape)

# Visualize results
selector.plot(top_n=10)
```

---

## How It Works

```
Data (X, y)
    │
    ▼
┌─────────────────────────────────────────────────┐
│         Parallel Computation (joblib)            │
│                                                  │
│  Correlation │ Permutation │  SHAP  │   LOFO    │
│  (optional)  │ (optional)  │(opt.)  │  (opt.)  │
└──────┬───────┴──────┬──────┴───┬────┴─────┬─────┘
       │              │          │           │
       └──────────────┴────┬─────┘───────────┘
                           ▼
           Aggregation (rank_mean / minmax_mean)
                    + Weighting
                           │
                           ▼
              Meta-Score Table (importance_df_)
                           │
                           ▼
              Best N Features (best_features_)
```

### Aggregation Detail (`rank_mean`)

1. Each method produces its own scores (e.g. SHAP values, correlation coefficients)
2. Features are **ranked** per method (1 = most important, N = least important)
3. The **weighted average** of all ranks is computed for each feature → `meta_score`
4. Lowest `meta_score` → best feature

---

## API Reference

### `ConsensusSelector(estimator, methods, aggregation, n_features_to_select, weights, n_jobs, scoring)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `estimator` | sklearn estimator | **Required** | The ML model to use. Will be trained internally for SHAP and Permutation. |
| `methods` | `list[str]` | `['correlation', 'permutation', 'shap']` | List of feature selection methods to run. |
| `aggregation` | `str` | `'rank_mean'` | Score combination strategy. Either `'rank_mean'` or `'minmax_mean'`. |
| `n_features_to_select` | `int` or `None` | `None` | Number of features to select. If `None`, all features are ranked. |
| `weights` | `dict` or `None` | `None` | Weight per method. e.g. `{'shap': 2.0, 'correlation': 0.5}`. |
| `n_jobs` | `int` | `-1` | Number of parallel jobs. `-1` uses all CPU cores. |
| `scoring` | `str` | `'roc_auc'` | Scoring metric used by the LOFO method. |

---

### Methods

#### `fit(X, y) → self`
Runs all feature selection computations and populates `importance_df_` and `best_features_`.

- `X`: `pd.DataFrame` or `np.ndarray` — input features
- `y`: `pd.Series` or `np.ndarray` — target variable

#### `transform(X) → pd.DataFrame or np.ndarray`
Returns the dataset filtered to only the selected features.

- Input type is preserved: DataFrame in → DataFrame out, ndarray in → ndarray out.

#### `fit_transform(X, y) → pd.DataFrame or np.ndarray`
Runs `fit(X, y)` followed by `transform(X)`.

#### `plot(top_n=15, title="Consensus Feature Selection Heatmap")`
Plots a heatmap showing each method's importance score for the top `top_n` features.

---

### Attributes (available after `fit()`)

| Attribute | Type | Description |
|-----------|------|-------------|
| `importance_df_` | `pd.DataFrame` | Full score table with each method and meta score as columns, sorted best-to-worst |
| `best_features_` | `list[str]` | Selected feature names in ranked order |
| `feature_names_` | `list[str]` | All feature names from training data |

---

## Supported Methods

### `'correlation'`
Computes the **absolute Pearson correlation** between each feature and the target. No model required — very fast. May miss non-linear relationships.

### `'permutation'`
Uses Scikit-learn's `permutation_importance`. Measures how much the model score drops when a feature's values are randomly shuffled. A large drop means the feature is important.

### `'shap'`
Computes SHAP (SHapley Additive exPlanations) values. Automatically selects the right explainer based on model type:
1. Tree-based models (XGBoost, LightGBM, RandomForest) → `TreeExplainer`
2. Linear models → `LinearExplainer`
3. All others → `KernelExplainer` (sampled to 100 rows for speed)

### `'lofo'`
LOFO (Leave One Feature Out) Importance: measures how much the cross-validation score changes when each feature is removed. The most interpretable method, but the slowest.

> ⚠️ LOFO can be slow on large datasets. Use the `scoring` parameter to choose an appropriate metric, or exclude it from `methods` if speed is a priority.

---

## Aggregation Strategies

### `rank_mean` (Recommended)
Uses rank information from each method. Not affected by scale differences between methods.

```
meta_score = weighted_average(rank_per_method)
```

- **Lower** `meta_score` → better feature
- If weights are provided, `numpy.average()` is used (ranking order is preserved)

### `minmax_mean`
Raw scores are normalized to 0-1 range, then averaged.

```
normalized = (score - min) / (max - min)
meta_score = weighted_average(normalized)
```

- **Higher** `meta_score` → better feature
- The original score magnitudes matter; dominant methods may overshadow others.

---

## Advanced Usage

### Custom Weights

```python
# Trust SHAP more, correlation less
custom_weights = {
    'shap': 2.0,
    'lofo': 1.5,
    'permutation': 1.0,
    'correlation': 0.5
}

selector = ConsensusSelector(
    estimator=model,
    methods=['correlation', 'permutation', 'shap', 'lofo'],
    weights=custom_weights,
    n_features_to_select=10
)
```

### Fast Mode (Without LOFO)

```python
selector = ConsensusSelector(
    estimator=model,
    methods=['correlation', 'shap'],
    aggregation='rank_mean',
    n_features_to_select=15,
    n_jobs=-1
)
```

### Ranking All Features

```python
# With n_features_to_select=None, all features are returned in ranked order
selector = ConsensusSelector(estimator=model)
selector.fit(X, y)

print(selector.importance_df_)    # Full score table
print(selector.best_features_)    # All features, best to worst
```

### Regression Problems

```python
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(random_state=42)

selector = ConsensusSelector(
    estimator=model,
    methods=['correlation', 'permutation', 'shap'],
    scoring='r2',           # appropriate metric for regression
    n_features_to_select=8
)
selector.fit(X_train, y_train)
```

---

## Sklearn Pipeline Integration

`ConsensusSelector` is fully compatible with Scikit-Learn Pipelines:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from consensusfs import ConsensusSelector

inner_model = RandomForestClassifier(n_estimators=50, random_state=42)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', ConsensusSelector(
        estimator=inner_model,
        methods=['correlation', 'shap'],
        n_features_to_select=10
    )),
    ('classifier', LogisticRegression())
])

pipe.fit(X_train, y_train)
print("Test Score:", pipe.score(X_test, y_test))
```

---

## Visualization

```python
# Call after fit()
selector.plot(top_n=15)

# With custom title
selector.plot(top_n=10, title="Project X — Feature Importance Map")
```

The heatmap shows each method's importance score for each feature, **normalized to 0-1** per column. Darker color = more important.

---

## Dependencies

| Library | Min Version | Purpose |
|---------|------------|---------|
| `scikit-learn` | ≥ 1.0.0 | BaseEstimator, permutation_importance |
| `pandas` | ≥ 1.0.0 | DataFrame operations |
| `numpy` | ≥ 1.18.0 | Numerical computations |
| `shap` | ≥ 0.40.0 | SHAP values |
| `lofo-importance` | ≥ 0.3.0 | LOFO computation |
| `joblib` | ≥ 1.0.0 | Parallel execution |
| `matplotlib` | ≥ 3.3.0 | Plotting |
| `seaborn` | ≥ 0.11.0 | Heatmap rendering |

---

## FAQ

**LOFO is too slow. What should I do?**
Remove `'lofo'` from the `methods` list. The other three methods are strong enough on their own.

**Which model should I use as the estimator?**
Tree-based models (RandomForest, XGBoost, LightGBM) give the fastest and most accurate SHAP results. Any Scikit-Learn compatible estimator will work, however.

**How do I choose `n_features_to_select`?**
Leave it as `None` first and inspect `importance_df_`. A good cutoff is where `meta_score` jumps significantly.

**Can I pass a NumPy array instead of a DataFrame?**
Yes. Both work. When a NumPy array is passed, feature names are automatically assigned as `feature_0`, `feature_1`, etc.

**What happens if I don't provide `weights`?**
All methods are weighted equally (1.0). For `rank_mean`, this means a simple average rank is computed.

**Does it re-train the model inside `fit()`?**
Yes. `ConsensusSelector.fit()` always **retrains** the estimator. This is the expected behavior when used inside a Pipeline.

---

## License

MIT License — free to use, modify, and distribute.

---

*Author: **Ulaş Taylan Met** — umet9711@gmail.com*
