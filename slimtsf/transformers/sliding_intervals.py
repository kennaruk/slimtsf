
from __future__ import annotations

"""
Deterministic sliding-window interval transformer for [n_cases, n_channels, n_timepoints] inputs.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union, Dict
import numpy as np
from joblib import Parallel, delayed

try:
    import pycatch22
    import antropy as ant
    _HAVE_EXTRAS = True
except ImportError:
    _HAVE_EXTRAS = False


# ----------------------------- Feature Logic -----------------------------

@dataclass(frozen=True)
class FeatureFunction:
    """
    A single feature extraction function.

    Parameters
    ----------
    name:
        Human-readable name that will appear in metadata and feature columns.
    function:
        Callable that accepts a 2D numpy.ndarray of shape [number_of_cases, window_size]
        and returns a 1D numpy.ndarray of shape [number_of_cases], containing the feature
        value for each case for that interval.
    """
    name: str
    function: Callable[[np.ndarray], np.ndarray]


def _compute_slope_for_segments(segments: np.ndarray) -> np.ndarray:
    """
    Compute the least-squares slope for each row in a 2D array of segments.

    Parameters
    ----------
    segments : np.ndarray
        Shape [number_of_cases, window_size].

    Returns
    -------
    slopes : np.ndarray
        Shape [number_of_cases].
    """
    if segments.ndim != 2:
        raise ValueError("segments must be 2D: [number_of_cases, window_size]")
    number_of_cases, window_size = segments.shape
    if window_size <= 1:
        return np.zeros(number_of_cases, dtype=float)

    # Center x and y for numerical stability
    x = np.arange(window_size, dtype=float)
    x_centered = x - x.mean()
    denominator = float(np.sum(x_centered ** 2))
    if denominator == 0.0:
        return np.zeros(number_of_cases, dtype=float)

    y_centered = segments - segments.mean(axis=1, keepdims=True)
    # Each row's slope = sum( (x - x_mean) * (y - y_mean) ) / sum( (x - x_mean)^2 )
    slopes = (y_centered * x_centered).sum(axis=1) / denominator
    return slopes.astype(float, copy=False)


def built_in_feature_functions(names: Sequence[str]) -> List[FeatureFunction]:
    """
    Create built-in feature functions by name.

    Supported names:
        - "mean"
        - "std" (population standard deviation, ddof=0)
        - "slope" (least-squares slope over index 0..window-1)

    Returns
    -------
    list of FeatureFunction
    """
    mapping: Dict[str, FeatureFunction] = {
        "mean": FeatureFunction(
            name="mean",
            function=lambda segments: segments.mean(axis=1).astype(float, copy=False),
        ),
        "std": FeatureFunction(
            name="std",
            function=lambda segments: segments.std(axis=1, ddof=0).astype(float, copy=False),
        ),
        "slope": FeatureFunction(
            name="slope",
            function=_compute_slope_for_segments,
        ),
        "median": FeatureFunction(
            name="median",
            function=lambda segments: np.median(segments, axis=1).astype(float, copy=False),
        ),
        "iqr": FeatureFunction(
            name="iqr",
            function=lambda segments: np.subtract(*np.percentile(segments, [75, 25], axis=1)).astype(float, copy=False),
        ),
        "min": FeatureFunction(
            name="min",
            function=lambda segments: np.min(segments, axis=1).astype(float, copy=False),
        ),
        "max": FeatureFunction(
            name="max",
            function=lambda segments: np.max(segments, axis=1).astype(float, copy=False),
        ),
    }

    if _HAVE_EXTRAS:
        mapping.update({
            "permutation_entropy": FeatureFunction(
                name="permutation_entropy",
                function=lambda segments: np.array([ant.perm_entropy(row, normalize=True) for row in segments], dtype=float),
            ),
            "sample_entropy": FeatureFunction(
                name="sample_entropy",
                function=lambda segments: np.array([ant.sample_entropy(row) for row in segments], dtype=float),
            ),
            "lempel_ziv_complexity": FeatureFunction(
                name="lempel_ziv_complexity",
                function=lambda segments: np.array([
                    ant.lziv_complexity(row > np.median(row), normalize=True) for row in segments
                ], dtype=float),
            ),
            "trev": FeatureFunction(
                name="trev",
                function=lambda segments: np.array([pycatch22.CO_trev_1_num(row.tolist()) for row in segments], dtype=float),
            ),
            "acf_first_min": FeatureFunction(
                name="acf_first_min",
                function=lambda segments: np.array([pycatch22.CO_FirstMin_ac(row.tolist()) for row in segments], dtype=float),
            ),
            "stretch_high": FeatureFunction(
                name="stretch_high",
                function=lambda segments: np.array([pycatch22.SB_MotifThree_quantile_hh(row.tolist()) for row in segments], dtype=float),
            ),
            "outlier_timing": FeatureFunction(
                name="outlier_timing",
                # The prompt ambiguous for positive vs negative but standard catch22 outlier timing is commonly the combination or positive
                # Let's use the positive outliers for mdrmd.
                function=lambda segments: np.array([pycatch22.DN_OutlierInclude_p_001_mdrmd(row.tolist()) for row in segments], dtype=float),
            ),
        })

    functions: List[FeatureFunction] = []
    for key in names:
        if key not in mapping:
            raise ValueError(f"Unsupported built-in feature name: '{key}'. "
                             f"Valid names are {sorted(list(mapping.keys()))}")
        functions.append(mapping[key])
    return functions


# ---------------------------- Transformer Class ----------------------------

class SlidingWindowIntervalTransformer:
    """
    Deterministic sliding-window interval transformer.

    Input
    -----
    X : numpy.ndarray of shape [number_of_cases, number_of_channels, number_of_timepoints]

    Output
    ------
    feature_matrix : numpy.ndarray of shape [number_of_cases, number_of_output_features]

    Parameters
    ----------
    window_sizes : Optional[Sequence[int]]
        Window sizes to apply. If None, window sizes are generated automatically as
        [T, T//2, T//4, ...] down to at least 2, where T is number_of_timepoints.
    window_step_ratio : float, default=0.5
        Step size as a ratio of the window size (must satisfy 0 < ratio <= 1).
        For example, 0.5 means 50% overlap (step = round(window_size * 0.5)).
    feature_functions : Optional[Sequence[Union[str, FeatureFunction]]], default=("mean","std","slope")
        List of feature functions to compute for each interval. Items can be:
            - strings for built-ins: "mean", "std", "slope"
            - FeatureFunction objects for custom logic
        Each function must map segments of shape [number_of_cases, window_size] to
        a 1D array [number_of_cases].
    number_of_jobs : int, default=1
        Number of parallel workers for interval computation (joblib). Use -1 for all CPUs.
    parallel_backend : Optional[str], default=None
        Joblib backend name (e.g., "loky", "threading"). If None, joblib chooses a sensible default.

    Notes
    -----
    - Deterministic: The enumeration order of intervals is fixed as:
        for channel in ascending order:
            for window_size in descending order:
                for start_index in ascending order:
                    yield interval
      Columns in the output feature matrix follow exactly that enumeration multiplied by the
      order of feature_functions.
    - Metadata: After fit(), the attribute `feature_metadata_` lists a dictionary per output column:
        { "channel_index": int, "start_index": int, "end_index": int, "window_size": int, "feature_name": str, "type": "interval" }
    """

    def __init__(
        self,
        window_sizes: Optional[Sequence[int]] = None,
        window_step_ratio: float = 0.5,
        feature_functions: Optional[Sequence[Union[str, FeatureFunction]]] = ("mean", "std", "slope"),
        number_of_jobs: int = 1,
        parallel_backend: Optional[str] = None,
    ) -> None:
        # Validate and store parameters
        if window_step_ratio <= 0.0 or window_step_ratio > 1.0:
            raise ValueError("window_step_ratio must be in (0, 1].")

        self.window_sizes = list(window_sizes) if window_sizes is not None else None
        self.window_step_ratio = float(window_step_ratio)
        self.number_of_jobs = int(number_of_jobs)
        self.parallel_backend = parallel_backend

        # Resolve feature functions
        if feature_functions is None:
            feature_functions = ("mean", "std", "slope")
        self.feature_functions: List[FeatureFunction] = []
        for item in feature_functions:
            if isinstance(item, str):
                # built-in mapping
                self.feature_functions.extend(built_in_feature_functions([item]))
            elif isinstance(item, FeatureFunction):
                self.feature_functions.append(item)
            else:
                raise TypeError("feature_functions must contain strings or FeatureFunction instances.")

        # Learned / fitted attributes
        self.number_of_channels_: Optional[int] = None
        self.number_of_timepoints_: Optional[int] = None
        self.interval_list_: List[Tuple[int, int, int, int]] = []  # (channel_index, start_index, end_index, window_size)
        self.feature_metadata_: List[Dict[str, Union[int, str]]] = []

    # -------------------- public API --------------------

    def fit(self, input_series: np.ndarray, target=None) -> "SlidingWindowIntervalTransformer":
        """
        Fit the transformer to infer interval layout (no randomness).

        Parameters
        ----------
        input_series : np.ndarray
            Shape [number_of_cases, number_of_channels, number_of_timepoints].

        Returns
        -------
        self
        """
        self._validate_input(input_series)
        number_of_cases, number_of_channels, number_of_timepoints = input_series.shape

        self.number_of_channels_ = number_of_channels
        self.number_of_timepoints_ = number_of_timepoints

        window_sizes_to_use = self._resolve_window_sizes(number_of_timepoints)

        # Prepare interval list and metadata in a deterministic order
        self.interval_list_.clear()
        self.feature_metadata_.clear()

        for channel_index in range(number_of_channels):
            for window_size in sorted(window_sizes_to_use, reverse=True):  # largest to smallest
                step_size = max(1, int(round(window_size * self.window_step_ratio)))
                last_start = number_of_timepoints - window_size
                if last_start < 0:
                    continue
                for start_index in range(0, last_start + 1, step_size):
                    end_index = start_index + window_size
                    self.interval_list_.append((channel_index, start_index, end_index, window_size))
                    for feature_function in self.feature_functions:
                        self.feature_metadata_.append({
                            "type": "interval",
                            "channel_index": channel_index,
                            "start_index": start_index,
                            "end_index": end_index,
                            "window_size": window_size,
                            "feature_name": feature_function.name,
                        })
                        
        # print("[debug] self.interval_list_:", self.interval_list_)
        # print("[debug] self.feature_metadata_:", self.feature_metadata_)
        return self

    def transform(self, input_series: np.ndarray) -> np.ndarray:
        """
        Transform the input series into a tabular feature matrix.

        Parameters
        ----------
        input_series : np.ndarray
            Shape [number_of_cases, number_of_channels, number_of_timepoints].

        Returns
        -------
        feature_matrix : np.ndarray
            Shape [number_of_cases, number_of_output_features]
        """
        self._validate_input(input_series)
        if not self.interval_list_:
            raise RuntimeError("Transformer must be fitted before calling transform().")
        number_of_cases = input_series.shape[0]

        # Helper to compute features for a single interval across all cases.
        def compute_features_for_interval(interval_tuple: Tuple[int, int, int, int]) -> np.ndarray:
            channel_index, start_index, end_index, window_size = interval_tuple
            segments = input_series[:, channel_index, start_index:end_index]  # [N, window_size]
            # Compute each feature function's output for this interval
            feature_columns = [feature_function.function(segments) for feature_function in self.feature_functions]
            interval_feature_block = np.vstack(feature_columns).T  # [N, number_of_feature_functions]
            return interval_feature_block

        if self.number_of_jobs == 1:
            # Deterministic single-thread path
            feature_blocks = [compute_features_for_interval(interval) for interval in self.interval_list_]
        else:
            # Parallel over intervals (the order of self.interval_list_ is preserved by joblib)
            feature_blocks = Parallel(n_jobs=self.number_of_jobs, backend=self.parallel_backend)(
                delayed(compute_features_for_interval)(interval) for interval in self.interval_list_
            )

        # Concatenate horizontally in the same deterministic order the intervals were enumerated
        feature_matrix = np.hstack(feature_blocks) if feature_blocks else np.zeros((number_of_cases, 0), dtype=float)
        return feature_matrix

    def fit_transform(self, input_series: np.ndarray, target=None) -> np.ndarray:
        """
        Fit to data, then transform.
        """
        return self.fit(input_series, target).transform(input_series)

    def get_feature_names_out(self) -> List[str]:
        """
        Return human-readable column names for the transformed feature matrix.
        """
        names: List[str] = []
        for meta in self.feature_metadata_:
            names.append(
                f"interval_channel_{meta['channel_index']}_start_{meta['start_index']}_end_{meta['end_index']}_{meta['feature_name']}"
            )
        return names

    # -------------------- utilities --------------------

    def _validate_input(self, input_series: np.ndarray) -> None:
        if not isinstance(input_series, np.ndarray):
            raise TypeError("input_series must be a numpy.ndarray.")
        if input_series.ndim != 3:
            raise ValueError("input_series must be 3D with shape [number_of_cases, number_of_channels, number_of_timepoints].")
        if input_series.shape[2] <= 1:
            raise ValueError("number_of_timepoints must be >= 2.")

    def _resolve_window_sizes(self, number_of_timepoints: int) -> List[int]:
        if self.window_sizes is not None and len(self.window_sizes) > 0:
            for w in self.window_sizes:
                if not isinstance(w, int) or w <= 1 or w > number_of_timepoints:
                    raise ValueError("Each window size must be an integer in [2, number_of_timepoints].")
            # Unique and sorted descending happens in fit()
            return list(self.window_sizes)
        # Automatic: start with full length, then repeatedly halve, down to at least 2
        sizes = []
        size = number_of_timepoints
        minimum_size = 2
        while size >= minimum_size:
            sizes.append(size)
            size = size // 2
        return sizes
