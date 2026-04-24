"""
slimtsf.classifier
~~~~~~~~~~~~~~~~~~
SlimTSFClassifier — a full Stage 1 + Stage 2 + Stage 3 pipeline
(SlidingWindowIntervalTransformer → IntervalStatsPoolingTransformer → RandomForestClassifier).
"""

from __future__ import annotations

import collections
import math
from typing import Optional, Sequence, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from slimtsf.transformers.sliding_intervals import FeatureFunction, SlidingWindowIntervalTransformer
from slimtsf.transformers.interval_stats_pooling import IntervalStatsPoolingTransformer


class SlimTSFClassifier:
    """
    Sliding-Window Multivariate Time-Series Forest Classifier.

    Bundles all three pipeline stages into a single sklearn-compatible estimator,
    with an optional Stage 4 for bootstrap feature selection.

        Stage 1 – ``SlidingWindowIntervalTransformer``
            Extracts sliding-window features (mean, std, slope by default)
            from a 3-D time-series array.

        Stage 2 – ``IntervalStatsPoolingTransformer``
            Pools the per-window features into global min / mean / max
            statistics per (channel, feature) group.

        Stage 3 – Bootstrap Feature Selection (Optional)
            Runs multiple Random Forest fits to identify and select only the
            most important and stable summary features.

        Stage 4 – ``RandomForestClassifier`` (scikit-learn)
            Classifies the final selected feature matrix.

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

    # --- Feature Configuration ---
    feature_mode : {"both", "interval", "pooled"}, default "both"
        Determines which features are passed downstream.
        If ``"both"``, interval features and pooled features are concatenated.
        If ``"interval"``, Stage 2 pooling is skipped and only raw interval features are passed.
        If ``"pooled"``, interval features are pooled in Stage 2 and then discarded.

    # --- Stage 3 (Bootstrap Selection) ---
    bootstrap : bool, default False
        Whether to perform bootstrap feature selection before final training.
    bootstrap_run : int, default 10
        Number of bootstrap iterations to run for feature ranking.
    top_rank : int, default 5
        Number of top features to select per bootstrap iteration for ranking.
    importance_method : {"gini", "permutation", "shap", "fisher", "anova-f"}, default "gini"
        Method to calculate feature importance during the bootstrap phase.
        Note: ``"fisher"`` and ``"anova-f"`` use the exact same calculation.

    # --- Stage 4 (Random Forest) ---
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
    verbose : int or bool, default False
        Controls the verbosity of pipeline execution and logs stage completion times.

    Attributes
    ----------
    stage1_ : SlidingWindowIntervalTransformer
        Fitted Stage 1 transformer.
    stage2_ : IntervalStatsPoolingTransformer
        Fitted Stage 2 transformer.
    stage3_ : RandomForestClassifier
        Fitted Stage 4 random forest.
    feature_indices_ : np.ndarray or None
        Indices of the selected features if ``bootstrap=True``; otherwise ``None``.
    classes_ : np.ndarray
        Unique class labels seen during ``fit``.
    n_features_in_ : int
        Number of features fed into the final random forest.

    Examples
    --------
    >>> import numpy as np
    >>> from slimtsf import SlimTSFClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((30, 2, 50))   # 30 cases, 2 channels, 50 time points
    >>> y = np.array([0] * 15 + [1] * 15)
    >>> clf = SlimTSFClassifier(bootstrap=True, n_estimators=50, random_state=0)
    >>> clf.fit(X, y)
    SlimTSFClassifier(bootstrap=True, n_estimators=50, random_state=0)
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
        feature_mode: str = "both",
        # Stage 3 (Bootstrap)
        bootstrap: bool = False,
        bootstrap_run: int = 10,
        top_rank: int = 5,
        importance_method: str = "gini",
        # Stage 4
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        class_weight: Optional[Union[str, dict]] = "balanced",
        random_state: Optional[int] = None,
        n_jobs: int = 1,
        # Stage 1 parallelism
        number_of_jobs: int = 1,
        parallel_backend: Optional[str] = None,
        verbose: Union[int, bool] = False,
    ) -> None:
        self.window_sizes = window_sizes
        self.window_step_ratio = window_step_ratio
        self.feature_functions = feature_functions
        self.aggregations = aggregations
        self.feature_mode = feature_mode
        self.bootstrap = bootstrap
        self.bootstrap_run = bootstrap_run
        self.top_rank = top_rank
        self.importance_method = importance_method
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.number_of_jobs = number_of_jobs
        self.parallel_backend = parallel_backend
        self.verbose = verbose

        # Fitted state — set by fit()
        self.stage1_: Optional[SlidingWindowIntervalTransformer] = None
        self.stage2_: Optional[IntervalStatsPoolingTransformer] = None
        self.stage3_: Optional[RandomForestClassifier] = None
        self.feature_indices_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.feature_selection_counts_: Optional[collections.Counter] = None

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
        if self.feature_mode not in ("interval", "pooled", "both"):
            raise ValueError(f"Unknown feature_mode: {self.feature_mode}")
        
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        # Stage 1 — sliding-window feature extraction
        if self.verbose:
            print("[SlimTSF] Starting Stage 1: Sliding Window feature extraction...")
        self.stage1_ = SlidingWindowIntervalTransformer(
            window_sizes=self.window_sizes,
            window_step_ratio=self.window_step_ratio,
            feature_functions=self.feature_functions,
            number_of_jobs=self.number_of_jobs,
            parallel_backend=self.parallel_backend,
            verbose=self.verbose,
        )
        interval_features = self.stage1_.fit_transform(X)

        # Stage 2 — stats pooling
        if self.verbose:
            print("[SlimTSF] Starting Stage 2: Interval Stats Pooling...")
        if self.feature_mode == "interval" or self.aggregations is None or len(self.aggregations) == 0:
            self.stage2_ = None
            if self.feature_mode == "pooled":
                raise ValueError("Cannot use feature_mode='pooled' without providing aggregations.")
            stage1_2_features = interval_features
        else:
            self.stage2_ = IntervalStatsPoolingTransformer(aggregations=self.aggregations)
            pooled_features = self.stage2_.fit_transform(
                interval_features,
                feature_metadata=self.stage1_.feature_metadata_,
            )
            if self.feature_mode == "pooled":
                stage1_2_features = pooled_features
            else:
                stage1_2_features = np.hstack((interval_features, pooled_features))

        # Stage 3 — Bootstrap Selection (Optional)
        if self.bootstrap:
            if self.verbose:
                print(f"[SlimTSF] Starting Stage 3: Bootstrap Feature Selection ({self.importance_method})...")
            selected_features = self._fit_bootstrap(stage1_2_features, y)
        else:
            selected_features = stage1_2_features
            self.feature_indices_ = None

        self.n_features_in_ = selected_features.shape[1]
        
        if self.bootstrap and self.verbose:
            print(f"[SlimTSF] Stage 3 Complete: Reduced feature matrix from {stage1_2_features.shape[1]} down to {self.n_features_in_} features.")

        # Stage 4 — final random forest
        if self.verbose:
            print("[SlimTSF] Starting Stage 4: Random Forest Classifier fitting...")
        self.stage3_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        self.stage3_.fit(selected_features, y)

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
        Return human-readable column names for the features fed into the final RF.

        Returns
        -------
        list of str
            One name per selected feature column.

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        """
        self._check_is_fitted()
        names_stage1 = self.stage1_.get_feature_names_out()
        if self.stage2_ is not None:
            names_stage2 = self.stage2_.get_feature_names_out()
            if self.feature_mode == "pooled":
                names = np.array(names_stage2)
            else:
                names = np.array(names_stage1 + names_stage2)
        else:
            names = np.array(names_stage1)
        
        if self.feature_indices_ is not None:
            names = names[self.feature_indices_]
        return names.tolist()

    def get_feature_selection_frequencies(self) -> dict[str, Union[int, float]]:
        """
        Return the frequency of selection for each feature across all bootstrap passes.
        
        Returns
        -------
        dict
            Mapping from feature name to the number of times it was selected in the top-k
            during the feature selection phase.
        """
        self._check_is_fitted()
        if not self.bootstrap:
            raise RuntimeError("Bootstrap feature selection was not enabled during fit.")
        if self.feature_selection_counts_ is None:
            return {}

        # Reconstruct all feature names (before selection)
        names_stage1 = self.stage1_.get_feature_names_out()
        if self.stage2_ is not None:
            names_stage2 = self.stage2_.get_feature_names_out()
            if self.feature_mode == "pooled":
                all_names = np.array(names_stage2)
            else:
                all_names = np.array(names_stage1 + names_stage2)
        else:
            all_names = np.array(names_stage1)

        return {all_names[idx]: count for idx, count in self.feature_selection_counts_.items()}

    def __repr__(self) -> str:
        params = (
            f"bootstrap={self.bootstrap!r}, "
            f"n_estimators={self.n_estimators!r}, "
            f"random_state={self.random_state!r}"
        )
        return f"SlimTSFClassifier({params})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_bootstrap(self, pooled_features: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Identify top stable features by running multiple RF passes or univariate statistical selections.
        """
        n_features = pooled_features.shape[1]
        n_to_select = math.ceil(math.log2(n_features)) if n_features > 0 else 0
        if n_to_select == 0:
            self.feature_indices_ = np.array([], dtype=int)
            return pooled_features[:, self.feature_indices_]

        # 1. Univariate Statistical Selection (Skip Model Constraints)
        if self.importance_method in ("fisher", "anova-f"):
            from sklearn.feature_selection import f_classif
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f_values, _ = f_classif(pooled_features, y)
            importances = np.nan_to_num(f_values, nan=0.0)
            sorted_idx = np.argsort(importances)[::-1]
            selected = sorted_idx[:n_to_select]
            self.feature_indices_ = np.sort(np.array(selected, dtype=int))
            self.feature_selection_counts_ = collections.Counter({idx: float(importances[idx]) for idx in range(len(importances))})
            if self.verbose:
                print(f"[SlimTSF] Univariate {self.importance_method}: Selected features -> {self.feature_indices_}")
            return pooled_features[:, self.feature_indices_]

        # 2. Ensemble Tree-based Bootstrap (Execute Parallel)
        import joblib
        from joblib import Parallel, delayed
        
        effective_n_jobs = joblib.cpu_count() if self.n_jobs == -1 else self.n_jobs
        outer_jobs = min(self.bootstrap_run, effective_n_jobs)
        inner_jobs = max(1, effective_n_jobs // outer_jobs)

        def _run_single_bootstrap(pass_idx: int) -> list[int]:
            # Initialize a fresh Random Forest for each bootstrap pass
            # Dynamically balance n_jobs preventing oversubscription while utilizing all hardware
            clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                class_weight=self.class_weight,
                random_state=self.random_state if self.random_state is None else self.random_state + pass_idx,
                n_jobs=inner_jobs,
            )
            clf.fit(pooled_features, y)
            
            if self.importance_method == "gini":
                importances = clf.feature_importances_
            elif self.importance_method == "permutation":
                from sklearn.inspection import permutation_importance
                result = permutation_importance(
                    clf, pooled_features, y, 
                    n_repeats=3, 
                    random_state=self.random_state if self.random_state is None else self.random_state + pass_idx,
                    n_jobs=inner_jobs
                )
                importances = result.importances_mean
            elif self.importance_method == "shap":
                import shap
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(pooled_features, check_additivity=False)
                
                if isinstance(shap_values, list):
                    importances = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
                elif len(shap_values.shape) == 3:
                    importances = np.abs(shap_values).mean(axis=0).mean(axis=1)
                else:
                    importances = np.abs(shap_values).mean(axis=0)
            else:
                raise ValueError(f"Unknown importance_method: {self.importance_method}")
            
            # Sort importances desc
            sorted_idx = np.argsort(importances)[::-1]
            return sorted_idx[:self.top_rank].tolist()

        # Joblib parallelism across entire bootstrapped passes simultaneously
        results = Parallel(n_jobs=outer_jobs, verbose=self.verbose)(
            delayed(_run_single_bootstrap)(i) for i in range(self.bootstrap_run)
        )
        
        top_features_all_runs = []
        for run_features in results:
            top_features_all_runs.extend(run_features)

        # Frequency count across passes
        counts = collections.Counter(top_features_all_runs)
        self.feature_selection_counts_ = counts
        
        selected = [idx for idx, _count in counts.most_common(n_to_select)]
        
        # Sort indices to keep feature order predictable
        self.feature_indices_ = np.sort(np.array(selected, dtype=int))
        
        if self.verbose:
            print(f"[SlimTSF] Selected {len(selected)} stable features from ensemble frequency counting.")
        
        return pooled_features[:, self.feature_indices_]

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted Stage 1 + Stage 2 (+ Stage 4 selection) to new data."""
        self._check_is_fitted()
        self._validate_X(X)
        interval_features = self.stage1_.transform(X)
        if self.stage2_ is not None:
            pooled_features = self.stage2_.transform(interval_features)
            if self.feature_mode == "pooled":
                stage1_2_features = pooled_features
            else:
                stage1_2_features = np.hstack((interval_features, pooled_features))
        else:
            stage1_2_features = interval_features
        
        if self.feature_indices_ is not None:
            return stage1_2_features[:, self.feature_indices_]
        return stage1_2_features

    def _check_is_fitted(self) -> None:
        if self.stage1_ is None or self.stage3_ is None:
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
