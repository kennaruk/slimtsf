import sys
from pathlib import Path

# Add v1 (package root) to path so "slimtsf" can be imported when script is run from anywhere
_v1_root = Path(__file__).resolve().parent.parent.parent
if str(_v1_root) not in sys.path:
    sys.path.insert(0, str(_v1_root))

import numpy as np

from slimtsf.transformers.sliding_intervals import (
    SlidingWindowIntervalTransformer,
    FeatureFunction,
)

# --- Helper functions for assertions ---
def assert_array_almost_equal(a, b, tol=1e-6):
    assert np.allclose(a, b, atol=tol), f"{a} != {b}"

# --- Tests ---

def test_single_case_mean_only():
    # One case, 1 channel, 6 timepoints: just [1, 2, 3, 4, 5, 6]
    X = np.arange(1, 7).reshape(1, 1, 6)
    sw = SlidingWindowIntervalTransformer(
        window_sizes=[3], window_step_ratio=1.0, feature_functions=["mean"]
    )
    features = sw.fit_transform(X) # [1,2,3], [4,5,6] 

    expected = np.array([
        [2, 5]  # each is mean([window])
    ])
    assert_array_almost_equal(features, expected)

def test_two_cases_mean():
    # 2 cases, 1 channel, 6 timepoints
    X = np.array([
        [[1, 2, 3, 4, 5, 6]],      # Case 1
        [[6, 5, 4, 3, 2, 1]],      # Case 2
    ])
    sw = SlidingWindowIntervalTransformer(
        window_sizes=[3], window_step_ratio=1.0, feature_functions=["mean"]
    )
    features = sw.fit_transform(X)
    expected = np.array([
        [2, 5],
        [5, 2],
    ])
    assert_array_almost_equal(features, expected)

def test_custom_feature_function():
    # Custom feature: sum
    X = np.array([[[1, 2, 3, 0, 0, 0]]])
    def sum_func(segments):
        return segments.sum(axis=1)
    custom_f = FeatureFunction(name="sum", function=sum_func)
    sw = SlidingWindowIntervalTransformer(
        window_sizes=[3], window_step_ratio=1.0, feature_functions=[custom_f]
    )
    features = sw.fit_transform(X)
    expected = np.array([[6, 0]])
    assert_array_almost_equal(features, expected)

def test_slope_and_mean_together():
    # One case, 1 channel, ascending values
    X = np.arange(1, 7).reshape(1, 1, 6)
    sw = SlidingWindowIntervalTransformer(
        window_sizes=[6], window_step_ratio=1.0, feature_functions=["mean", "slope"]
    )
    features = sw.fit_transform(X)
    # The only window is [1,2,3,4,5,6], mean=3.5, slope=1.0
    expected = np.array([[3.5, 1.0]])
    assert_array_almost_equal(features, expected)

def test_two_channels_two_windows_mean_sum():
    X = np.array([
        [[1, 2, 3, 4]],
        [[4, 3, 2, 1]]
    ])

    print("X:", X, X.shape)

    def sum_func(segments):
        return segments.sum(axis=1)
    custom_f = FeatureFunction(name="sum", function=sum_func)

    sw = SlidingWindowIntervalTransformer(
        window_sizes=[4, 2], window_step_ratio=0.5, feature_functions=["mean", custom_f]
    )
    features = sw.fit_transform(X) 
    # [1,2,3,4] [1,2], [2,3], [3,4]
    # [4,3,2,1] [4,3], [3,2], [2,1]

    print("features:", features)
    expected = np.array([
        # mean, sum, mean, sum
        [2.5, 10, 1.5, 3, 2.5, 5, 3.5, 7],
        [2.5, 10, 3.5, 7, 2.5, 5, 1.5, 3]
    ])

    assert features.shape == expected.shape, f"Output shape {features.shape}, expected {expected.shape}"
    assert_array_almost_equal(features, expected)
