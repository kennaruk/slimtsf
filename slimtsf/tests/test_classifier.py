"""
Tests for SlimTSFClassifier — Stage 1 + Stage 2 + Stage 3 end-to-end.
"""

import sys
from pathlib import Path

_v1_root = Path(__file__).resolve().parent.parent.parent
if str(_v1_root) not in sys.path:
    sys.path.insert(0, str(_v1_root))

import numpy as np
import pytest

from slimtsf import (
    SlimTSFClassifier,
    SlidingWindowIntervalTransformer,
    IntervalStatsPoolingTransformer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_dataset():
    """20 cases, 2 channels, 30 time points, binary labels."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 2, 30))
    y = np.array([0] * 10 + [1] * 10)
    return X, y


@pytest.fixture
def fitted_clf(small_dataset):
    X, y = small_dataset
    clf = SlimTSFClassifier(n_estimators=10, random_state=0)
    clf.fit(X, y)
    return clf, X, y


# ---------------------------------------------------------------------------
# Import checks
# ---------------------------------------------------------------------------

class TestPublicImports:
    """Verify the public API is importable from the top-level package."""

    def test_import_classifier(self):
        from slimtsf import SlimTSFClassifier
        assert SlimTSFClassifier is not None

    def test_import_stage1(self):
        from slimtsf import SlidingWindowIntervalTransformer
        assert SlidingWindowIntervalTransformer is not None

    def test_import_stage2(self):
        from slimtsf import IntervalStatsPoolingTransformer
        assert IntervalStatsPoolingTransformer is not None

    def test_version_exists(self):
        import slimtsf
        assert hasattr(slimtsf, "__version__")
        assert isinstance(slimtsf.__version__, str)


# ---------------------------------------------------------------------------
# Fit / Predict
# ---------------------------------------------------------------------------

class TestFitPredict:
    def test_fit_returns_self(self, small_dataset):
        X, y = small_dataset
        clf = SlimTSFClassifier(n_estimators=10, random_state=0)
        result = clf.fit(X, y)
        assert result is clf

    def test_predict_shape(self, fitted_clf):
        clf, X, _ = fitted_clf
        preds = clf.predict(X)
        assert preds.shape == (20,)

    def test_predict_labels_are_valid(self, fitted_clf):
        clf, X, _ = fitted_clf
        preds = clf.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, fitted_clf):
        clf, X, _ = fitted_clf
        proba = clf.predict_proba(X)
        assert proba.shape == (20, 2)

    def test_predict_proba_sums_to_one(self, fitted_clf):
        clf, X, _ = fitted_clf
        proba = clf.predict_proba(X)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_classes_attribute(self, fitted_clf):
        clf, _, _ = fitted_clf
        assert hasattr(clf, "classes_")
        assert list(clf.classes_) == [0, 1]

    def test_n_features_in(self, fitted_clf):
        clf, _, _ = fitted_clf
        assert hasattr(clf, "n_features_in_")
        assert isinstance(clf.n_features_in_, int)
        assert clf.n_features_in_ > 0


# ---------------------------------------------------------------------------
# Predict on unseen data (critical: must not refit)
# ---------------------------------------------------------------------------

class TestPredictOnTestData:
    def test_predict_different_n_cases(self, fitted_clf):
        """Predict on a different number of cases than training."""
        clf, _, _ = fitted_clf
        rng = np.random.default_rng(99)
        X_test = rng.standard_normal((5, 2, 30))  # 5 cases, same shape otherwise
        preds = clf.predict(X_test)
        assert preds.shape == (5,)

    def test_predict_proba_different_n_cases(self, fitted_clf):
        clf, _, _ = fitted_clf
        rng = np.random.default_rng(99)
        X_test = rng.standard_normal((7, 2, 30))
        proba = clf.predict_proba(X_test)
        assert proba.shape == (7, 2)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_random_state_same_predictions(self, small_dataset):
        X, y = small_dataset
        clf1 = SlimTSFClassifier(n_estimators=10, random_state=7)
        clf2 = SlimTSFClassifier(n_estimators=10, random_state=7)
        preds1 = clf1.fit(X, y).predict(X)
        preds2 = clf2.fit(X, y).predict(X)
        np.testing.assert_array_equal(preds1, preds2)


# ---------------------------------------------------------------------------
# Configurability
# ---------------------------------------------------------------------------

class TestConfigurability:
    def test_custom_window_sizes(self, small_dataset):
        X, y = small_dataset
        clf = SlimTSFClassifier(window_sizes=[4, 8], n_estimators=5, random_state=0)
        clf.fit(X, y)
        assert clf.predict(X).shape == (20,)

    def test_custom_aggregations(self, small_dataset):
        X, y = small_dataset
        clf = SlimTSFClassifier(aggregations=("min", "max"), n_estimators=5, random_state=0)
        clf.fit(X, y)
        assert clf.predict(X).shape == (20,)

    def test_custom_feature_functions(self, small_dataset):
        X, y = small_dataset
        clf = SlimTSFClassifier(
            feature_functions=["mean"],
            n_estimators=5,
            random_state=0,
        )
        clf.fit(X, y)
        assert clf.predict(X).shape == (20,)

    def test_single_channel(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((15, 1, 20))
        y = np.array([0] * 8 + [1] * 7)
        clf = SlimTSFClassifier(n_estimators=5, random_state=0)
        clf.fit(X, y)
        assert clf.predict(X).shape == (15,)


# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------

class TestFeatureNamesOut:
    def test_returns_list_of_strings(self, fitted_clf):
        clf, _, _ = fitted_clf
        names = clf.get_feature_names_out()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_length_matches_n_features_in(self, fitted_clf):
        clf, _, _ = fitted_clf
        names = clf.get_feature_names_out()
        assert len(names) == clf.n_features_in_

    def test_raises_before_fit(self):
        clf = SlimTSFClassifier()
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.get_feature_names_out()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_predict_before_fit_raises(self):
        clf = SlimTSFClassifier()
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict(np.zeros((5, 2, 20)))

    def test_non_array_raises(self, small_dataset):
        X, y = small_dataset
        clf = SlimTSFClassifier(n_estimators=5, random_state=0)
        with pytest.raises(TypeError):
            clf.fit([[1, 2, 3]], y[:1])

    def test_2d_array_raises(self, small_dataset):
        X, y = small_dataset
        clf = SlimTSFClassifier(n_estimators=5, random_state=0)
        with pytest.raises(ValueError, match="3-D"):
            clf.fit(X[:, 0, :], y)  # 2D — wrong shape
