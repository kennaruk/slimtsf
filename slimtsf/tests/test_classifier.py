"""
Tests for SlimTSFClassifier — Stage 1 + Stage 2 + Stage 3 end-to-end.
"""

import math
import sys
from pathlib import Path

_v1_root = Path(__file__).resolve().parent.parent.parent
if str(_v1_root) not in sys.path:
    sys.path.insert(0, str(_v1_root))

import numpy as np
import pytest

from slimtsf import (
    SlimTSFClassifier,
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

    def test_stage2_concatenates_stage1(self, small_dataset):
        X, y = small_dataset
        clf = SlimTSFClassifier(window_sizes=[5], n_estimators=5, random_state=0)
        clf.fit(X, y)
        
        # Manually compute Stage 1 and Stage 2 feature counts
        interval_features = clf.stage1_.transform(X)
        pooled_features = clf.stage2_.transform(interval_features)
        
        # Verify concatenation: The model gets BOTH sets of features
        expected_total_features = interval_features.shape[1] + pooled_features.shape[1]
        assert clf.n_features_in_ == expected_total_features


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

    def test_skip_stage_2(self, small_dataset):
        X, y = small_dataset
        clf = SlimTSFClassifier(aggregations=None, n_estimators=5, random_state=0)
        clf.fit(X, y)
        assert clf.stage2_ is None
        assert clf.predict(X).shape == (20,)
        
    def test_skip_stage_2_empty_tuple(self, small_dataset):
        X, y = small_dataset
        clf = SlimTSFClassifier(aggregations=(), n_estimators=5, random_state=0)
        clf.fit(X, y)
        assert clf.stage2_ is None
        assert clf.predict(X).shape == (20,)

    def test_feature_mode_interval(self, small_dataset):
        X, y = small_dataset
        clf = SlimTSFClassifier(feature_mode="interval", n_estimators=5, random_state=0)
        clf.fit(X, y)
        assert clf.stage2_ is None
        assert clf.predict(X).shape == (20,)
        
        interval_features = clf.stage1_.transform(X)
        assert clf.n_features_in_ == interval_features.shape[1]
        
    def test_feature_mode_pooled(self, small_dataset):
        X, y = small_dataset
        clf = SlimTSFClassifier(feature_mode="pooled", n_estimators=5, random_state=0)
        clf.fit(X, y)
        assert clf.stage2_ is not None
        assert clf.predict(X).shape == (20,)
        
        interval_features = clf.stage1_.transform(X)
        pooled_features = clf.stage2_.transform(interval_features)
        assert clf.n_features_in_ == pooled_features.shape[1]

    def test_feature_mode_both(self, small_dataset):
        X, y = small_dataset
        clf = SlimTSFClassifier(feature_mode="both", n_estimators=5, random_state=0)
        clf.fit(X, y)
        assert clf.stage2_ is not None
        assert clf.predict(X).shape == (20,)
        
        interval_features = clf.stage1_.transform(X)
        pooled_features = clf.stage2_.transform(interval_features)
        assert clf.n_features_in_ == interval_features.shape[1] + pooled_features.shape[1]

    def test_feature_mode_invalid_raises(self, small_dataset):
        X, y = small_dataset
        clf = SlimTSFClassifier(feature_mode="invalid_mode", n_estimators=5, random_state=0)
        with pytest.raises(ValueError, match="feature_mode"):
            clf.fit(X, y)


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
# Bootstrap Feature Selection (Stage 4)
# ---------------------------------------------------------------------------

class TestBootstrap:
    def test_bootstrap_reduces_feature_count(self, small_dataset):
        X, y = small_dataset
        
        # 1. No bootstrap
        clf_no = SlimTSFClassifier(bootstrap=False, n_estimators=5, random_state=0)
        clf_no.fit(X, y)
        n_features_no = clf_no.n_features_in_
        
        # 2. With bootstrap
        clf_yes = SlimTSFClassifier(bootstrap=True, bootstrap_run=3, top_rank=2, n_estimators=5, random_state=0)
        clf_yes.fit(X, y)
        n_features_yes = clf_yes.n_features_in_
        
        # Bootstrap should pick at most ceil(log2(N)) features.
        # It could be less if fewer unique features were selected across bootstrap runs.
        assert n_features_yes < n_features_no
        assert n_features_yes <= math.ceil(math.log2(n_features_no))

    def test_bootstrap_indices_stored(self, small_dataset):
        X, y = small_dataset
        clf = SlimTSFClassifier(bootstrap=True, bootstrap_run=2, n_estimators=5, random_state=0)
        clf.fit(X, y)
        assert clf.feature_indices_ is not None
        assert len(clf.feature_indices_) == clf.n_features_in_

    def test_bootstrap_predict_works(self, small_dataset):
        X, y = small_dataset
        clf = SlimTSFClassifier(bootstrap=True, bootstrap_run=2, n_estimators=5, random_state=0)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_bootstrap_importance_method_gini(self, small_dataset):
        X, y = small_dataset
        clf = SlimTSFClassifier(bootstrap=True, bootstrap_run=2, top_rank=2, importance_method="gini", n_estimators=5, random_state=0)
        clf.fit(X, y)
        assert clf.feature_indices_ is not None

    def test_bootstrap_importance_method_permutation(self, small_dataset):
        X, y = small_dataset
        clf = SlimTSFClassifier(bootstrap=True, bootstrap_run=2, top_rank=2, importance_method="permutation", n_estimators=5, random_state=0)
        clf.fit(X, y)
        assert clf.feature_indices_ is not None

    def test_bootstrap_importance_method_shap(self, small_dataset):
        pytest.importorskip("shap")
        X, y = small_dataset
        clf = SlimTSFClassifier(bootstrap=True, bootstrap_run=2, top_rank=2, importance_method="shap", n_estimators=5, random_state=0)
        clf.fit(X, y)
        assert clf.feature_indices_ is not None

    def test_bootstrap_importance_method_fisher(self, small_dataset):
        X, y = small_dataset
        clf = SlimTSFClassifier(bootstrap=True, bootstrap_run=2, top_rank=2, importance_method="fisher", n_estimators=5, random_state=0)
        clf.fit(X, y)
        assert clf.feature_indices_ is not None
        assert len(clf.feature_indices_) > 0

    def test_bootstrap_importance_method_anova_f(self, small_dataset):
        X, y = small_dataset
        clf = SlimTSFClassifier(bootstrap=True, bootstrap_run=2, top_rank=2, importance_method="anova-f", n_estimators=5, random_state=0)
        clf.fit(X, y)
        assert clf.feature_indices_ is not None
        assert len(clf.feature_indices_) > 0

    def test_bootstrap_invalid_importance_method(self, small_dataset):
        X, y = small_dataset
        clf = SlimTSFClassifier(bootstrap=True, bootstrap_run=1, importance_method="invalid", n_estimators=5)
        with pytest.raises(ValueError, match="Unknown importance_method"):
            clf.fit(X, y)

    def test_feature_selection_frequencies(self, small_dataset):
        X, y = small_dataset
        clf = SlimTSFClassifier(bootstrap=True, bootstrap_run=3, top_rank=2, n_estimators=5, random_state=0)
        clf.fit(X, y)
        
        freq = clf.get_feature_selection_frequencies()
        assert isinstance(freq, dict)
        assert len(freq) > 0
        # Check that values are integers (from ensemble counting)
        assert all(isinstance(v, int) for v in freq.values())
        # The sum of all counts should be exactly top_rank * bootstrap_run = 2 * 3 = 6
        assert sum(freq.values()) == 6

    def test_feature_selection_frequencies_fisher(self, small_dataset):
        X, y = small_dataset
        clf = SlimTSFClassifier(bootstrap=True, importance_method="fisher", n_estimators=5, random_state=0)
        clf.fit(X, y)
        
        freq = clf.get_feature_selection_frequencies()
        assert isinstance(freq, dict)
        assert len(freq) > 0
        # Fisher returns F-scores which are floats
        assert all(isinstance(v, float) for v in freq.values())
        
        # Check that asking for freq without bootstrap raises an error
        clf_no = SlimTSFClassifier(bootstrap=False, n_estimators=5, random_state=0)
        clf_no.fit(X, y)
        with pytest.raises(RuntimeError, match="Bootstrap feature selection was not enabled"):
            clf_no.get_feature_selection_frequencies()


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
