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


class TestFeatureFunction:
    
    def test_numpy_features(self):
        # A simple array to test mapping
        X = np.arange(1, 13).reshape(1, 1, 12)
        # 12 elements: 1 to 12. window size 12
        sw = SlidingWindowIntervalTransformer(
            window_sizes=[12], window_step_ratio=1.0, 
            feature_functions=["median", "iqr", "min", "max"]
        )
        features = sw.fit_transform(X)
        # Expected:
        # median: 6.5
        # iqr: 75th percentile is 9.25, 25th percentile is 3.75, so 9.25 - 3.75 = 5.5
        expected = np.array([[6.5, 5.5, 1.0, 12.0]])
        assert_array_almost_equal(features, expected)
        
    def test_antropy_features(self):
        # We need realistic non-flat data to avoid entropy being zero or trivial
        np.random.seed(42)
        X = np.random.randn(1, 1, 100)
        sw = SlidingWindowIntervalTransformer(
            window_sizes=[100], window_step_ratio=1.0, 
            feature_functions=["permutation_entropy", "sample_entropy", "lempel_ziv_complexity"]
        )
        features = sw.fit_transform(X)
        assert features.shape == (1, 3)
        # Just assert they run without error and give valid (non-NaN) outputs
        expected = np.array([[0.9990777207672243, 1.845826690498331, 1.2623326760571978]])
        assert_array_almost_equal(features, expected)
        
    def test_pycatch22_features(self):
        np.random.seed(42)
        X = np.random.randn(1, 1, 100)
        sw = SlidingWindowIntervalTransformer(
            window_sizes=[100], window_step_ratio=1.0, 
            feature_functions=["trev", "acf_first_min", "stretch_high", "outlier_timing"]
        )
        features = sw.fit_transform(X)
        expected = np.array([[-0.3527509032403569, 2, 2.1934541579986027, 0.10000000000000009]])
        assert_array_almost_equal(features, expected)

    def test_invalid_feature_name(self):
        import pytest
        with pytest.raises(ValueError, match="Unsupported built-in feature name"):
            sw = SlidingWindowIntervalTransformer(
                window_sizes=[10], window_step_ratio=1.0, 
                feature_functions=["not_a_real_feature"]
            )
