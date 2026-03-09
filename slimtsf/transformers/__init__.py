"""
slimtsf.transformers
~~~~~~~~~~~~~~~~~~~~
Sliding-window and pooling transformers for multivariate time-series.
"""

from slimtsf.transformers.sliding_intervals import (
    SlidingWindowIntervalTransformer,
    FeatureFunction,
    built_in_feature_functions,
)
from slimtsf.transformers.interval_stats_pooling import IntervalStatsPoolingTransformer

__all__ = [
    "SlidingWindowIntervalTransformer",
    "FeatureFunction",
    "built_in_feature_functions",
    "IntervalStatsPoolingTransformer",
]
