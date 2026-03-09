"""
Tests for IntervalStatsPoolingTransformer (Stage 2).

Structure
---------
Part A – Unit tests for IntervalStatsPoolingTransformer in isolation:
    - Hand-crafted interval_features + metadata so expected values are
      trivial to compute by hand.

Part B – Integration tests with SlidingWindowIntervalTransformer (Stage 1):
    - Build X → Stage 1 → Stage 2 → assert final shape and values.
    - One-channel and two-channel variants.
"""

import sys
from pathlib import Path

# Let the test run from anywhere (pytest, direct python invocation, etc.)
_v1_root = Path(__file__).resolve().parent.parent.parent
if str(_v1_root) not in sys.path:
    sys.path.insert(0, str(_v1_root))

import numpy as np
import pytest

from slimtsf.transformers.interval_stats_pooling import IntervalStatsPoolingTransformer
from slimtsf.transformers.sliding_intervals import SlidingWindowIntervalTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def almost_equal(a: np.ndarray, b: np.ndarray, tol: float = 1e-7) -> None:
    """Assert two arrays are element-wise close."""
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    assert np.allclose(a, b, atol=tol), (
        f"Values differ:\n  got:      {a}\n  expected: {b}"
    )


def make_metadata(
    channel_index: int,
    feature_name: str,
    n_windows: int,
    window_size: int = 2,
) -> list:
    """
    Build a minimal feature_metadata list for ``n_windows`` interval columns,
    all belonging to the same (channel_index, feature_name) group.
    """
    return [
        {
            "type": "interval",
            "channel_index": channel_index,
            "start_index": i * window_size,
            "end_index": (i + 1) * window_size,
            "window_size": window_size,
            "feature_name": feature_name,
        }
        for i in range(n_windows)
    ]


# ===========================================================================
# Part A – Unit tests (isolated, hand-crafted data)
# ===========================================================================


class TestTwoRowsSingleFeature:
    """
    2 cases, 1 channel, 3 windows, 1 feature (mean).

    interval_features:
        case 0: [1.0, 2.0, 3.0]   → min=1  mean=2  max=3
        case 1: [4.0, 6.0, 2.0]   → min=2  mean=4  max=6

    Each row is pooled independently (rows are never mixed).
    """

    def setup_method(self):
        self.interval_features = np.array(
            [
                [1.0, 2.0, 3.0],  # case 0
                [4.0, 6.0, 2.0],  # case 1
            ]
        )
        self.metadata = make_metadata(channel_index=0, feature_name="mean", n_windows=3)
        self.pooling = IntervalStatsPoolingTransformer(aggregations=("min", "mean", "max"))

    def test_shape(self):
        pooled = self.pooling.fit_transform(
            self.interval_features, feature_metadata=self.metadata
        )
        # 1 group × 3 aggregations = 3 output columns
        assert pooled.shape == (2, 3)

    def test_values(self):
        pooled = self.pooling.fit_transform(
            self.interval_features, feature_metadata=self.metadata
        )
        expected = np.array(
            [
                [1.0, 2.0, 3.0],   # case 0: min, mean, max of [1,2,3]
                [2.0, 4.0, 6.0],   # case 1: min, mean, max of [4,6,2]
            ]
        )
        almost_equal(pooled, expected)

    def test_cases_pooled_independently(self):
        """Changing one case must not affect the other."""
        pooled = self.pooling.fit_transform(
            self.interval_features, feature_metadata=self.metadata
        )
        # case 0's min is 1, not influenced by case 1's values
        assert pooled[0, 0] == pytest.approx(1.0)
        # case 1's max is 6, not influenced by case 0's values
        assert pooled[1, 2] == pytest.approx(6.0)


class TestThreeRowsSingleFeature:
    """
    3 cases, 1 channel, 4 windows, 1 feature (std).

    interval_features:
        case 0: [0.0, 0.0, 0.0, 0.0]  → min=0 mean=0 max=0
        case 1: [1.0, 2.0, 3.0, 4.0]  → min=1 mean=2.5 max=4
        case 2: [5.0, 3.0, 5.0, 3.0]  → min=3 mean=4 max=5
    """

    def setup_method(self):
        self.interval_features = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],   # case 0
                [1.0, 2.0, 3.0, 4.0],   # case 1
                [5.0, 3.0, 5.0, 3.0],   # case 2
            ]
        )
        self.metadata = make_metadata(channel_index=0, feature_name="std", n_windows=4)

    def test_shape(self):
        pooling = IntervalStatsPoolingTransformer(aggregations=("min", "mean", "max"))
        pooled = pooling.fit_transform(self.interval_features, feature_metadata=self.metadata)
        assert pooled.shape == (3, 3)

    def test_values(self):
        pooling = IntervalStatsPoolingTransformer(aggregations=("min", "mean", "max"))
        pooled = pooling.fit_transform(self.interval_features, feature_metadata=self.metadata)
        expected = np.array(
            [
                [0.0, 0.0, 0.0],         # case 0
                [1.0, 2.5, 4.0],         # case 1
                [3.0, 4.0, 5.0],         # case 2
            ]
        )
        almost_equal(pooled, expected)

    def test_case0_all_zeros(self):
        """All-zero windows should produce all-zero pooled values."""
        pooling = IntervalStatsPoolingTransformer(aggregations=("min", "mean", "max"))
        pooled = pooling.fit_transform(self.interval_features, feature_metadata=self.metadata)
        assert np.all(pooled[0] == 0.0)


class TestTwoFeaturesOneChannel:
    """
    1 case, 1 channel, 2 windows, 2 features (mean and std).

    Column layout (same as SlidingWindowIntervalTransformer output):
        col 0: win0_mean  col 1: win0_std  col 2: win1_mean  col 3: win1_std

    Groups formed:
        (ch=0, "mean") → cols [0, 2] → values [1.5, 3.5]  → min=1.5, mean=2.5, max=3.5
        (ch=0, "std")  → cols [1, 3] → values [0.5, 0.5]  → min=0.5, mean=0.5, max=0.5
    """

    def setup_method(self):
        # Manually lay out columns the same way Stage 1 does:
        # per interval: [mean, std, mean, std]
        self.interval_features = np.array([[1.5, 0.5, 3.5, 0.5]])  # 1 case, 4 cols
        self.metadata = [
            {"type": "interval", "channel_index": 0, "start_index": 0, "end_index": 2, "window_size": 2, "feature_name": "mean"},
            {"type": "interval", "channel_index": 0, "start_index": 0, "end_index": 2, "window_size": 2, "feature_name": "std"},
            {"type": "interval", "channel_index": 0, "start_index": 2, "end_index": 4, "window_size": 2, "feature_name": "mean"},
            {"type": "interval", "channel_index": 0, "start_index": 2, "end_index": 4, "window_size": 2, "feature_name": "std"},
        ]

    def test_shape(self):
        pooling = IntervalStatsPoolingTransformer(aggregations=("min", "mean", "max"))
        pooled = pooling.fit_transform(self.interval_features, feature_metadata=self.metadata)
        # 2 groups × 3 aggregations = 6 columns
        assert pooled.shape == (1, 6)

    def test_values(self):
        pooling = IntervalStatsPoolingTransformer(aggregations=("min", "mean", "max"))
        pooled = pooling.fit_transform(self.interval_features, feature_metadata=self.metadata)
        expected = np.array([[1.5, 2.5, 3.5, 0.5, 0.5, 0.5]])
        almost_equal(pooled, expected)

    def test_features_not_mixed(self):
        """mean-group and std-group must be pooled separately."""
        pooling = IntervalStatsPoolingTransformer(aggregations=("min", "mean", "max"))
        pooled = pooling.fit_transform(self.interval_features, feature_metadata=self.metadata)
        # First 3 cols → mean group (min=1.5, mean=2.5, max=3.5)
        almost_equal(pooled[:, :3], np.array([[1.5, 2.5, 3.5]]))
        # Last 3 cols  → std group  (min=0.5, mean=0.5, max=0.5)
        almost_equal(pooled[:, 3:], np.array([[0.5, 0.5, 0.5]]))


class TestMinOnlyAggregation:
    """Use a non-default aggregations config: only 'min'."""

    def test_shape_and_values(self):
        interval_features = np.array([[4.0, 2.0, 8.0]])  # 1 case, 3 windows
        metadata = make_metadata(channel_index=0, feature_name="slope", n_windows=3)
        pooling = IntervalStatsPoolingTransformer(aggregations=("min",))
        pooled = pooling.fit_transform(interval_features, feature_metadata=metadata)
        assert pooled.shape == (1, 1)
        assert pooled[0, 0] == pytest.approx(2.0)


class TestInvalidInputs:
    """Guard-rail checks."""

    def test_bad_aggregation(self):
        with pytest.raises(ValueError, match="Unsupported aggregation"):
            IntervalStatsPoolingTransformer(aggregations=("median",))

    def test_empty_aggregations(self):
        with pytest.raises(ValueError, match="At least one"):
            IntervalStatsPoolingTransformer(aggregations=())

    def test_metadata_length_mismatch(self):
        pooling = IntervalStatsPoolingTransformer()
        features = np.ones((2, 4))
        bad_meta = make_metadata(0, "mean", n_windows=3)  # 3 ≠ 4
        with pytest.raises(ValueError, match="feature_metadata has"):
            pooling.fit(features, feature_metadata=bad_meta)

    def test_no_metadata_on_first_fit(self):
        pooling = IntervalStatsPoolingTransformer()
        with pytest.raises(ValueError, match="feature_metadata must be provided"):
            pooling.fit(np.ones((2, 3)))

    def test_transform_before_fit(self):
        pooling = IntervalStatsPoolingTransformer()
        with pytest.raises(RuntimeError, match="Call fit"):
            pooling.transform(np.ones((2, 3)))


# ===========================================================================
# Part B – Integration tests: Stage 1 → Stage 2
# ===========================================================================


class TestIntegrationOneChannel:
    """
    One channel, window_size=2, step=1.0 (non-overlapping), features=[mean, std].

    Time series: X = [[1, 2, 3, 4]]  (1 case, 1 channel, 4 timepoints)

    Stage 1 produces two windows of size 2:
        win0 = [1, 2]  → mean=1.5, std=0.5
        win1 = [3, 4]  → mean=3.5, std=0.5

    Stage 1 column order (channel → window → feature):
        col 0: win0_mean = 1.5
        col 1: win0_std  = 0.5
        col 2: win1_mean = 3.5
        col 3: win1_std  = 0.5

    Stage 2 groups:
        (ch=0, "mean") → [1.5, 3.5] → min=1.5, mean=2.5, max=3.5
        (ch=0, "std")  → [0.5, 0.5] → min=0.5, mean=0.5, max=0.5

    Expected output: [[1.5, 2.5, 3.5, 0.5, 0.5, 0.5]]
    """

    def setup_method(self):
        X = np.arange(1, 5).reshape(1, 1, 4).astype(float)
        sw = SlidingWindowIntervalTransformer(
            window_sizes=[2],
            window_step_ratio=1.0,
            feature_functions=["mean", "std"],
        )
        self.interval_features = sw.fit_transform(X)
        self.metadata = sw.feature_metadata_
        self.expected = np.array([[1.5, 2.5, 3.5, 0.5, 0.5, 0.5]])

    def test_stage1_shape(self):
        # Sanity: 1 case, 2 windows × 2 features = 4 columns
        assert self.interval_features.shape == (1, 4)

    def test_stage2_shape(self):
        pooling = IntervalStatsPoolingTransformer(aggregations=("min", "mean", "max"))
        pooled = pooling.fit_transform(
            self.interval_features, feature_metadata=self.metadata
        )
        # 2 feature groups × 3 aggregations = 6 columns
        assert pooled.shape == (1, 6)

    def test_stage2_values(self):
        pooling = IntervalStatsPoolingTransformer(aggregations=("min", "mean", "max"))
        pooled = pooling.fit_transform(
            self.interval_features, feature_metadata=self.metadata
        )
        almost_equal(pooled, self.expected)

    def test_feature_names_out(self):
        pooling = IntervalStatsPoolingTransformer(aggregations=("min", "mean", "max"))
        pooling.fit(self.interval_features, feature_metadata=self.metadata)
        names = pooling.get_feature_names_out()
        assert len(names) == 6
        assert names[0] == "pooled_channel_0_mean_min"
        assert names[1] == "pooled_channel_0_mean_mean"
        assert names[2] == "pooled_channel_0_mean_max"
        assert names[3] == "pooled_channel_0_std_min"


class TestIntegrationTwoChannels:
    """
    Two channels, window_size=2, step=1.0 (non-overlapping), feature=[mean].

    Time series:
        X shape [1, 2, 4]
        ch0 = [1, 2, 3, 4]
        ch1 = [10, 20, 30, 40]

    Stage 1 column order (channel asc → window asc → feature):
        col 0: ch0_win0_mean = mean([1,2])   = 1.5
        col 1: ch0_win1_mean = mean([3,4])   = 3.5
        col 2: ch1_win0_mean = mean([10,20]) = 15.0
        col 3: ch1_win1_mean = mean([30,40]) = 35.0

    Stage 2 groups (channels are NEVER mixed):
        (ch=0, "mean") → [1.5, 3.5]   → min=1.5,  mean=2.5,  max=3.5
        (ch=1, "mean") → [15.0, 35.0] → min=15.0, mean=25.0, max=35.0

    Expected output: [[1.5, 2.5, 3.5, 15.0, 25.0, 35.0]]
    """

    def setup_method(self):
        X = np.array(
            [[[1.0, 2.0, 3.0, 4.0],
              [10.0, 20.0, 30.0, 40.0]]]
        )  # shape [1, 2, 4]
        sw = SlidingWindowIntervalTransformer(
            window_sizes=[2],
            window_step_ratio=1.0,
            feature_functions=["mean"],
        )
        self.interval_features = sw.fit_transform(X)
        self.metadata = sw.feature_metadata_

    def test_stage1_shape(self):
        # 1 case, 2 channels × 2 windows × 1 feature = 4 columns
        assert self.interval_features.shape == (1, 4)

    def test_stage2_shape(self):
        pooling = IntervalStatsPoolingTransformer(aggregations=("min", "mean", "max"))
        pooled = pooling.fit_transform(self.interval_features, feature_metadata=self.metadata)
        # 2 channel-groups × 3 aggregations = 6 columns
        assert pooled.shape == (1, 6)

    def test_stage2_values(self):
        pooling = IntervalStatsPoolingTransformer(aggregations=("min", "mean", "max"))
        pooled = pooling.fit_transform(self.interval_features, feature_metadata=self.metadata)
        expected = np.array([[1.5, 2.5, 3.5, 15.0, 25.0, 35.0]])
        almost_equal(pooled, expected)

    def test_channels_are_independent(self):
        """ch1 windows have no influence on ch0 pooled values and vice-versa."""
        pooling = IntervalStatsPoolingTransformer(aggregations=("min", "mean", "max"))
        pooled = pooling.fit_transform(self.interval_features, feature_metadata=self.metadata)
        ch0_pooled = pooled[:, :3]   # first 3 cols = ch0 group
        ch1_pooled = pooled[:, 3:]   # last 3 cols  = ch1 group
        almost_equal(ch0_pooled, np.array([[1.5, 2.5, 3.5]]))
        almost_equal(ch1_pooled, np.array([[15.0, 25.0, 35.0]]))

    def test_feature_names_contain_correct_channels(self):
        pooling = IntervalStatsPoolingTransformer(aggregations=("min", "mean", "max"))
        pooling.fit(self.interval_features, feature_metadata=self.metadata)
        names = pooling.get_feature_names_out()
        assert all("channel_0" in n for n in names[:3])
        assert all("channel_1" in n for n in names[3:])


class TestIntegrationMultipleCasesOneCh:
    """
    Two cases, 1 channel – verifies rows are pooled independently end-to-end.

    X:
        case 0: ch0 = [1, 2, 3, 4]
        case 1: ch0 = [4, 3, 2, 1]

    Stage 2 (mean only):
        case 0: (ch0,"mean") → [1.5, 3.5] → min=1.5, mean=2.5, max=3.5
        case 1: (ch0,"mean") → [3.5, 1.5] → min=1.5, mean=2.5, max=3.5
    """

    def setup_method(self):
        X = np.array(
            [
                [[1.0, 2.0, 3.0, 4.0]],   # case 0
                [[4.0, 3.0, 2.0, 1.0]],   # case 1
            ]
        )  # shape [2, 1, 4]
        sw = SlidingWindowIntervalTransformer(
            window_sizes=[2],
            window_step_ratio=1.0,
            feature_functions=["mean"],
        )
        self.interval_features = sw.fit_transform(X)
        self.metadata = sw.feature_metadata_

    def test_shape(self):
        pooling = IntervalStatsPoolingTransformer(aggregations=("min", "mean", "max"))
        pooled = pooling.fit_transform(self.interval_features, feature_metadata=self.metadata)
        assert pooled.shape == (2, 3)

    def test_values(self):
        pooling = IntervalStatsPoolingTransformer(aggregations=("min", "mean", "max"))
        pooled = pooling.fit_transform(self.interval_features, feature_metadata=self.metadata)
        expected = np.array(
            [
                [1.5, 2.5, 3.5],   # case 0
                [1.5, 2.5, 3.5],   # case 1 (same pooled values, different arrangement)
            ]
        )
        almost_equal(pooled, expected)
