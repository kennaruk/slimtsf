"""
slimtsf — Sliding-Window Multivariate Time-Series Forest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A minimal, scikit-learn–compatible library for multi-scale sliding-window
feature extraction and classification of multivariate time-series data.

Quick Start
-----------
>>> import numpy as np
>>> from slimtsf import SlimTSFClassifier
>>> X = np.random.randn(20, 2, 50)   # 20 cases, 2 channels, 50 time points
>>> y = np.array([0] * 10 + [1] * 10)
>>> clf = SlimTSFClassifier(n_estimators=100, random_state=0)
>>> clf.fit(X, y)
>>> clf.predict(X)
"""

try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("slimtsf")
    except PackageNotFoundError:
        __version__ = "0.0.0.dev"
except ImportError:
    __version__ = "0.0.0.dev"

from slimtsf.classifier import SlimTSFClassifier
from slimtsf.transformers import (
    SlidingWindowIntervalTransformer,
    IntervalStatsPoolingTransformer,
    FeatureFunction,
    built_in_feature_functions,
)

__all__ = [
    # Full pipeline
    "SlimTSFClassifier",
    # Individual transformers (for composable use)
    "SlidingWindowIntervalTransformer",
    "IntervalStatsPoolingTransformer",
    # Feature function helpers
    "FeatureFunction",
    "built_in_feature_functions",
    # Package metadata
    "__version__",
]
