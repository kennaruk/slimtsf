"""
slimtsf.classifier
~~~~~~~~~~~~~~~~~~
SlimTSFClassifier — a full Stage 1 + Stage 2 + Stage 3 pipeline
(SlidingWindowIntervalTransformer → IntervalStatsPoolingTransformer → RandomForestClassifier).
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from slimtsf.transformers.sliding_intervals import FeatureFunction, SlidingWindowIntervalTransformer
from slimtsf.transformers.interval_stats_pooling import IntervalStatsPoolingTransformer


class SlimTSFClassifier:
    """
    Sliding-Window Multivariate Time-Series Forest Classifier.

    Bundles all three pipeline stages into a single sklearn-compatible estimator:

        Stage 1 – ``SlidingWindowIntervalTransformer``
            Extracts sliding-window features (mean, std, slope by default)
            from a 3-D time-series array.

        Stage 2 – ``IntervalStatsPoolingTransformer``
            Pools the per-window features into global min / mean / max
            statistics per (channel, feature) group.

        Stage 3 – ``RandomForestClassifier`` (scikit-learn)
            Classifies the compact pooled feature matrix.

    Parameters
    ----------
    # --- Stage 1 ---
    window_sizes : sequence of int or None, default None
        Window sizes for the sliding-window step.  If ``None``, sizes are
        derived automatically as ``[T, T//2, T//4, …]`` down to 2, where
        ``T`` is the number of time points.
    window_step_ratio : float, default 0.5
        Step size as a fraction of the window size (0 < ratio <= 1).
        0.5 → 50 % overlap between consecutive windows.
    feature_functions : sequence of str or FeatureFunction, default ("mean", "std", "slope")
        Features computed for each window.  Built-in strings: ``"mean"``,
        ``"std"``, ``"slope"``.  Pass a ``FeatureFunction`` for custom logic.

    # --- Stage 2 ---
    aggregations : sequence of str, default ("min", "mean", "max")
        Statistics to pool across the window dimension for each
        (channel, feature) group.  Supported: ``"min"``, ``"mean"``, ``"max"``.

    # --- Stage 3 (Random Forest) ---
    n_estimators : int, default 200
        Number of trees in the random forest.
    max_depth : int or None, default None
        Maximum tree depth.  ``None`` means trees grow until all leaves are
        pure or contain fewer than ``min_samples_split`` samples.
    class_weight : str, dict or None, default "balanced"
        Weights associated with classes.  ``"balanced"`` adjusts weights
        inversely to class frequency — useful for imbalanced data.
    random_state : int or None, default None
        Seed for reproducibility.
    n_jobs : int, default 1
        Number of parallel jobs for the random forest (``-1`` = all CPUs).

    # --- Parallelism (Stage 1) ---
    number_of_jobs : int, default 1
        Number of parallel workers for interval computation in Stage 1.
    parallel_backend : str or None, default None
        Joblib backend for Stage 1 (e.g. ``"loky"``, ``"threading"``).

    Attributes
    ----------
    stage1_ : SlidingWindowIntervalTransformer
        Fitted Stage 1 transformer.
    stage2_ : IntervalStatsPoolingTransformer
        Fitted Stage 2 transformer.
    stage3_ : RandomForestClassifier
        Fitted Stage 3 random forest.
    classes_ : np.ndarray
        Unique class labels seen during ``fit``.
    n_features_in_ : int
        Number of pooled features fed into the random forest.

    Examples
    --------
    >>> import numpy as np
    >>> from slimtsf import SlimTSFClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((30, 2, 50))   # 30 cases, 2 channels, 50 time points
    >>> y = np.array([0] * 15 + [1] * 15)
    >>> clf = SlimTSFClassifier(n_estimators=50, random_state=0)
    >>> clf.fit(X, y)
    SlimTSFClassifier(n_estimators=50, random_state=0)
    >>> preds = clf.predict(X)
    >>> preds.shape
    (30,)
    """

    def __init__(
        self,
        # Stage 1
        window_sizes: Optional[Sequence[int]] = None,
        window_step_ratio: float = 0.5,
        feature_functions: Optional[Sequence[Union[str, FeatureFunction]]] = (
            "mean", "std", "slope"
        ),
        # Stage 2
        aggregations: Sequence[str] = ("min", "mean", "max"),
        # Stage 3
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        class_weight: Optional[Union[str, dict]] = "balanced",
        random_state: Optional[int] = None,
        n_jobs: int = 1,
        # Stage 1 parallelism
        number_of_jobs: int = 1,
        parallel_backend: Optional[str] = None,
    ) -> None:
        self.window_sizes = window_sizes
        self.window_step_ratio = window_step_ratio
        self.feature_functions = feature_functions
        self.aggregations = aggregations
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.number_of_jobs = number_of_jobs
        self.parallel_backend = parallel_backend

        # Fitted state — set by fit()
        self.stage1_: Optional[SlidingWindowIntervalTransformer] = None
        self.stage2_: Optional[IntervalStatsPoolingTransformer] = None
        self.stage3_: Optional[RandomForestClassifier] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SlimTSFClassifier":
        """
        Fit the full pipeline on training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_cases, n_channels, n_timepoints)
            Multivariate time-series training data.
        y : np.ndarray, shape (n_cases,)
            Class labels.

        Returns
        -------
        self
        """
        self._validate_X(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        # Stage 1 — sliding-window feature extraction
        self.stage1_ = SlidingWindowIntervalTransformer(
            window_sizes=self.window_sizes,
            window_step_ratio=self.window_step_ratio,
            feature_functions=self.feature_functions,
            number_of_jobs=self.number_of_jobs,
            parallel_backend=self.parallel_backend,
        )
        interval_features = self.stage1_.fit_transform(X)

        # Stage 2 — stats pooling
        self.stage2_ = IntervalStatsPoolingTransformer(aggregations=self.aggregations)
        pooled_features = self.stage2_.fit_transform(
            interval_features,
            feature_metadata=self.stage1_.feature_metadata_,
        )

        self.n_features_in_ = pooled_features.shape[1]

        # Stage 3 — random forest
        self.stage3_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.stage3_.fit(pooled_features, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input time-series.

        Parameters
        ----------
        X : np.ndarray, shape (n_cases, n_channels, n_timepoints)

        Returns
        -------
        y_pred : np.ndarray, shape (n_cases,)
        """
        self._check_is_fitted()
        return self.stage3_.predict(self._transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input time-series.

        Parameters
        ----------
        X : np.ndarray, shape (n_cases, n_channels, n_timepoints)

        Returns
        -------
        proba : np.ndarray, shape (n_cases, n_classes)
        """
        self._check_is_fitted()
        return self.stage3_.predict_proba(self._transform(X))

    def get_feature_names_out(self) -> list[str]:
        """
        Return human-readable column names for the pooled features fed into the RF.

        Returns
        -------
        list of str
            One name per pooled feature column.

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        """
        self._check_is_fitted()
        return self.stage2_.get_feature_names_out()

    def __repr__(self) -> str:
        params = (
            f"n_estimators={self.n_estimators!r}, "
            f"random_state={self.random_state!r}"
        )
        return f"SlimTSFClassifier({params})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted Stage 1 + Stage 2 to new data (no re-fitting)."""
        self._check_is_fitted()
        self._validate_X(X)
        interval_features = self.stage1_.transform(X)
        return self.stage2_.transform(interval_features)

    def _check_is_fitted(self) -> None:
        if self.stage1_ is None or self.stage2_ is None or self.stage3_ is None:
            raise RuntimeError(
                "This SlimTSFClassifier instance is not fitted yet. "
                "Call 'fit' before using this estimator."
            )

    @staticmethod
    def _validate_X(X: np.ndarray) -> None:
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy.ndarray.")
        if X.ndim != 3:
            raise ValueError(
                "X must be 3-D with shape (n_cases, n_channels, n_timepoints). "
                f"Got shape {X.shape}."
            )
