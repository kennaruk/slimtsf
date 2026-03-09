from __future__ import annotations

"""
Stage 2 – Interval-level stats pooling transformer.

Input
-----
interval_features : np.ndarray, shape [n_cases, n_interval_features]
    The 2-D feature matrix from Stage 1 (``SlidingWindowIntervalTransformer``).

Output
------
pooled_features : np.ndarray, shape [n_cases, n_pooled_features]
    where ``n_pooled_features`` =
        n_unique_channel_feature_pairs × len(aggregations)

Column order in the output
--------------------------
Groups are sorted by ``(channel_index, feature_name)``; within each
group the aggregations appear in the order given by ``aggregations``.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


class IntervalStatsPoolingTransformer:
    """
    Pool interval-level features into global min / mean / max statistics.

    Parameters
    ----------
    aggregations : sequence of str, default ``("min", "mean", "max")``
        Which statistics to compute over the window dimension for each
        ``(channel_index, feature_name)`` group.  Supported values:
        ``"min"``, ``"mean"``, ``"max"``.
        Order here controls the order of output columns within each group.

    Notes
    -----
    The transformer expects per-column metadata produced by
    ``SlidingWindowIntervalTransformer.feature_metadata_``.  Each entry
    is a dict with at least::

        {
            "type": "interval",   # must be "interval" to be grouped
            "channel_index": int,
            "feature_name": str,
        }

    Pass this list as ``feature_metadata`` on the first call to ``fit()``
    or ``fit_transform()``.
    """

    def __init__(
        self,
        aggregations: Sequence[str] = ("min", "mean", "max"),
    ) -> None:
        if not isinstance(aggregations, Iterable):
            raise TypeError("aggregations must be an iterable of strings.")

        _valid = {"min", "mean", "max"}
        cleaned: List[str] = []
        for agg in aggregations:
            if agg not in _valid:
                raise ValueError(
                    f"Unsupported aggregation '{agg}'. "
                    f"Valid options are {sorted(_valid)}."
                )
            cleaned.append(str(agg))

        if not cleaned:
            raise ValueError("At least one aggregation must be specified.")

        # Immutable tuple for clear repr / hashing
        self.aggregations: Tuple[str, ...] = tuple(cleaned)

        # Fitted attributes (set by fit())
        self.feature_metadata_: Optional[List[Dict[str, object]]] = None
        # List of ((channel_index, feature_name), [col_idx, ...]) in sorted order
        self._groups_: Optional[List[Tuple[Tuple[int, str], List[int]]]] = None
        # Metadata describing each output column (after fit)
        self.output_feature_metadata_: Optional[List[Dict[str, object]]] = None
        self._fitted: bool = False

    def fit(
        self,
        interval_features: np.ndarray,
        feature_metadata: Optional[Sequence[Dict[str, object]]] = None,
        target=None,
    ) -> "IntervalStatsPoolingTransformer":
        """
        Infer column groupings from metadata (no data-dependent parameters).

        Parameters
        ----------
        interval_features : np.ndarray, shape [n_cases, n_interval_features]
            Used only to validate the metadata length.
        feature_metadata : list of dict, optional
            One dict per column in ``interval_features``.  Required on the
            first call; may be omitted on subsequent calls if
            ``self.feature_metadata_`` is already set.
        target :
            Ignored.  Present for pipeline compatibility.

        Returns
        -------
        self
        """
        if not isinstance(interval_features, np.ndarray):
            raise TypeError("interval_features must be a numpy.ndarray.")
        if interval_features.ndim != 2:
            raise ValueError("interval_features must be a 2-D array.")

        _, n_features = interval_features.shape

        # Accept or validate metadata
        if feature_metadata is not None:
            meta_list = list(feature_metadata)
            if len(meta_list) != n_features:
                raise ValueError(
                    f"feature_metadata has {len(meta_list)} entries but "
                    f"interval_features has {n_features} columns."
                )
            self.feature_metadata_ = meta_list
        else:
            if self.feature_metadata_ is None:
                raise ValueError(
                    "feature_metadata must be provided on the first call to "
                    "fit() (or pass it to fit_transform())."
                )
            if len(self.feature_metadata_) != n_features:
                raise ValueError(
                    "Stored feature_metadata_ length does not match "
                    "interval_features column count."
                )

        # ----------------------------------------------------------------
        # Build groups:
        # splits columns into window_count chunks of width count_skip,
        # then strides by n_features within each chunk to pick channel f.
        # Here we reach the same grouping by reading per-column metadata.
        # ----------------------------------------------------------------
        groups_dict: Dict[Tuple[int, str], List[int]] = {}
        for col_idx, meta in enumerate(self.feature_metadata_):  # type: ignore[arg-type]
            if not isinstance(meta, dict):
                raise TypeError("Each feature_metadata entry must be a dict.")
            if meta.get("type") != "interval":
                # Non-interval columns are skipped (forward-compatibility)
                continue
            if "channel_index" not in meta or "feature_name" not in meta:
                raise KeyError(
                    "Each interval metadata dict must have 'channel_index' "
                    "and 'feature_name'."
                )
            key: Tuple[int, str] = (int(meta["channel_index"]), str(meta["feature_name"]))
            groups_dict.setdefault(key, []).append(col_idx)

        # Sort deterministically: channel first, then feature name
        self._groups_ = [
            (key, groups_dict[key])
            for key in sorted(groups_dict.keys(), key=lambda k: (k[0], k[1]))
        ]

        # Build output column metadata
        output_meta: List[Dict[str, object]] = []
        for (ch_idx, feat_name), _ in self._groups_:
            for agg in self.aggregations:
                output_meta.append(
                    {
                        "type": "pooled",
                        "channel_index": ch_idx,
                        "feature_name": feat_name,
                        "aggregation": agg,
                    }
                )
        self.output_feature_metadata_ = output_meta
        self._fitted = True
        return self

    def transform(self, interval_features: np.ndarray) -> np.ndarray:
        """
        Pool interval features into global statistics.

        Parameters
        ----------
        interval_features : np.ndarray, shape [n_cases, n_interval_features]

        Returns
        -------
        pooled_features : np.ndarray, shape [n_cases, n_pooled_features]
        """
        if not self._fitted or self._groups_ is None:
            raise RuntimeError("Call fit() before transform().")
        if not isinstance(interval_features, np.ndarray):
            raise TypeError("interval_features must be a numpy.ndarray.")
        if interval_features.ndim != 2:
            raise ValueError("interval_features must be 2-D.")

        n_cases, n_features = interval_features.shape
        if self.feature_metadata_ is None or len(self.feature_metadata_) != n_features:
            raise ValueError(
                "interval_features column count does not match what was seen at fit time."
            )

        pooled_columns: List[np.ndarray] = []

        for _group_key, col_indices in self._groups_:
            # sub_matrix: [n_cases, n_windows_for_this_group]
            # Each column is the feature value computed on one sliding window.
            sub_matrix = interval_features[:, col_indices]

            if sub_matrix.shape[1] == 0:
                continue  # defensive; should not happen

            for agg in self.aggregations:
                # axis=1  →  collapse across the window dimension,
                # keepdims=True  →  preserve shape [n_cases, 1] for hstack
                if agg == "min":
                    pooled = sub_matrix.min(axis=1, keepdims=True)
                elif agg == "mean":
                    pooled = sub_matrix.mean(axis=1, keepdims=True)
                elif agg == "max":
                    pooled = sub_matrix.max(axis=1, keepdims=True)
                else:
                    raise RuntimeError(f"Unexpected aggregation '{agg}'.")

                pooled_columns.append(pooled.astype(float, copy=False))

        if not pooled_columns:
            return np.zeros((n_cases, 0), dtype=float)

        return np.hstack(pooled_columns)

    def fit_transform(
        self,
        interval_features: np.ndarray,
        feature_metadata: Optional[Sequence[Dict[str, object]]] = None,
        target=None,
    ) -> np.ndarray:
        """
        Convenience: fit then transform in one call.
        """
        return self.fit(
            interval_features,
            feature_metadata=feature_metadata,
            target=target,
        ).transform(interval_features)

    def get_feature_names_out(self) -> List[str]:
        """
        Return a human-readable column name for each output feature.

        Format: ``pooled_channel_<ch>_<feature_name>_<aggregation>``

        Example: ``pooled_channel_0_mean_min``
        """
        if self.output_feature_metadata_ is None:
            raise RuntimeError("Call fit() before get_feature_names_out().")
        return [
            f"pooled_channel_{m['channel_index']}_{m['feature_name']}_{m['aggregation']}"
            for m in self.output_feature_metadata_
        ]
