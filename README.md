# slimtsf · Sliding-Window Multivariate Time-Series Forest

[![PyPI version](https://badge.fury.io/py/slimtsf.svg)](https://pypi.org/project/slimtsf/)
[![CI](https://github.com/kennaruk/slimtsf/actions/workflows/ci.yml/badge.svg)](https://github.com/kennaruk/slimtsf/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A minimal, **scikit-learn–compatible** library for classifying multivariate time-series data using multi-scale sliding-window feature extraction.

---

## Install

```bash
pip install slimtsf
```

---

## Quick Start

### Full pipeline (recommended)

```python
import numpy as np
from slimtsf import SlimTSFClassifier

# X: (n_cases, n_channels, n_timepoints)  — 3-D numpy array
X_train = np.random.randn(100, 3, 120)
y_train = np.array([0] * 50 + [1] * 50)

clf = SlimTSFClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

X_test = np.random.randn(20, 3, 120)
predictions  = clf.predict(X_test)        # shape (20,)
probabilities = clf.predict_proba(X_test) # shape (20, 2)
```

### Transformers only (composable use)

```python
from slimtsf import SlidingWindowIntervalTransformer, IntervalStatsPoolingTransformer

# Stage 1 — extract sliding-window features
stage1 = SlidingWindowIntervalTransformer(
    window_sizes=[8, 16, 32],      # or None for auto
    window_step_ratio=0.5,
    feature_functions=["mean", "std", "slope"],
)
interval_features = stage1.fit_transform(X_train)  # (n_cases, n_interval_features)

# Stage 2 — pool across windows
stage2 = IntervalStatsPoolingTransformer(aggregations=("min", "mean", "max"))
pooled = stage2.fit_transform(
    interval_features,
    feature_metadata=stage1.feature_metadata_,     # wires Stage 1 → Stage 2
)  # (n_cases, n_pooled_features)
```

### Use with scikit-learn tools

Because `SlimTSFClassifier` exposes fitted stage attributes, you can access the underlying sklearn RF and use it with standard sklearn utilities:

```python
from sklearn.model_selection import cross_val_score

# Fit first, then use sklearn metrics on transformed data
clf.fit(X_train, y_train)
Xt = clf.stage2_.transform(clf.stage1_.transform(X_train))

scores = cross_val_score(clf.stage3_, Xt, y_train, cv=5)
print(scores.mean())
```

---

## How It Works

```
3-D time-series X  (n_cases, n_channels, n_timepoints)
    │
    ▼  Stage 1 — SlidingWindowIntervalTransformer
    │  Slide windows of multiple sizes across each channel.
    │  Compute mean / std / slope per window.
    │  Output: 2-D matrix  (n_cases, n_interval_features)
    │
    ▼  Stage 2 — IntervalStatsPoolingTransformer
    │  For each (channel, feature) group,
    │  pool across windows: min / mean / max.
    │  Output: 2-D compact matrix  (n_cases, n_pooled_features)
    │  *Note: These pooled features are concatenated with Stage 1 features.*
    │
    ▼  Stage 3 — Bootstrap Feature Selection (Optional)
    │  Run multiple Random Forest passes to rank and select the top
    │  most stable features (log2 of total features).
    │  Output: 2-D refined matrix  (n_cases, n_selected_features)
    │
    ▼  Stage 4 — RandomForestClassifier (scikit-learn)
       Classify the final selected feature matrix.
       Output: predicted labels / probabilities
```

---

## API Reference

### `SlimTSFClassifier`

| Parameter           | Type                         | Default                  | Description                                                          |
| ------------------- | ---------------------------- | ------------------------ | -------------------------------------------------------------------- |
| `window_sizes`      | `list[int] \| None`          | `None`                   | Window sizes. Auto if `None` (`[T, T//2, …]`).                       |
| `window_step_ratio` | `float`                      | `0.5`                    | Step = ratio × window size.                                          |
| `feature_functions` | `list[str\|FeatureFunction]` | `("mean","std","slope")` | Per-window features.                                                 |
| `aggregations`      | `list[str] \| None`          | `("min","mean","max")`   | Pooling statistics across windows. Pass `None` to skip Stage 2.      |
| `bootstrap`         | `bool`                       | `False`                  | Run multi-pass feature selection before final RF.                    |
| `bootstrap_run`     | `int`                        | `10`                     | Number of passes for feature ranking.                                |
| `top_rank`          | `int`                        | `5`                      | Top features to select per pass.                                     |
| `importance_method` | `str`                        | `"gini"`                 | Method for feature calculation: `"gini"`, `"permutation"`, `"shap"`. |
| `n_estimators`      | `int`                        | `200`                    | Number of RF trees.                                                  |
| `max_depth`         | `int\|None`                  | `None`                   | Max tree depth.                                                      |
| `class_weight`      | `str\|dict\|None`            | `"balanced"`             | RF class weighting.                                                  |
| `random_state`      | `int\|None`                  | `None`                   | Reproducibility seed.                                                |
| `n_jobs`            | `int`                        | `1`                      | Parallel jobs for RF (`-1` = all CPUs).                              |

**Methods:** `fit(X, y)` · `predict(X)` · `predict_proba(X)` · `get_feature_names_out()`

**Fitted attributes:** `stage1_` · `stage2_` · `stage3_` · `classes_` · `n_features_in_`

---

### `SlidingWindowIntervalTransformer`

**Input:** `X` — shape `(n_cases, n_channels, n_timepoints)`  
**Output:** 2-D feature matrix `(n_cases, n_interval_features)`

**Methods:** `fit(X)` · `transform(X)` · `fit_transform(X)` · `get_feature_names_out()`  
**Fitted attributes:** `feature_metadata_` · `interval_list_`

---

### `IntervalStatsPoolingTransformer`

**Input:** 2-D interval feature matrix from Stage 1 + `feature_metadata`  
**Output:** 2-D pooled feature matrix `(n_cases, n_pooled_features)`

**Methods:** `fit(X, feature_metadata)` · `transform(X)` · `fit_transform(X, feature_metadata)` · `get_feature_names_out()`

---

## Custom Feature Functions

```python
from slimtsf import FeatureFunction, SlidingWindowIntervalTransformer
import numpy as np

# A custom feature: interquartile range
iqr = FeatureFunction(
    name="iqr",
    function=lambda seg: np.percentile(seg, 75, axis=1) - np.percentile(seg, 25, axis=1),
)

transformer = SlidingWindowIntervalTransformer(feature_functions=["mean", iqr])
```

---

## Versioning

This project follows [Semantic Versioning](https://semver.org/) and [Conventional Commits](https://www.conventionalcommits.org/):

| Commit prefix                 | Effect                |
| ----------------------------- | --------------------- |
| `fix:`                        | patch release (0.1.x) |
| `feat:`                       | minor release (0.x.0) |
| `feat!:` / `BREAKING CHANGE:` | major release (x.0.0) |
| `docs:` `chore:` `test:`      | no release            |

---

## Development

```bash
git clone https://github.com/kennaruk/slimtsf.git
cd slimtsf
pip install -e ".[dev]"
pytest -v
```

---

## License

MIT — see [LICENSE](LICENSE).
