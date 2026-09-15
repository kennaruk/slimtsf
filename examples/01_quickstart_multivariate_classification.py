"""
SlimTSF: Multivariate Time-Series Classification Quickstart
============================================================

This example demonstrates how to:
1. Simulate a realistic 3-channel multivariate sensor dataset.
2. Train a `SlimTSFClassifier` using multi-scale sliding windows,
   interval pooling, and bootstrap feature stability selection.
3. Evaluate classification accuracy and inspect the top selected features.
4. Use individual transformers (`SlidingWindowIntervalTransformer` and
   `IntervalStatsPoolingTransformer`) in a modular pipeline.
"""

import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from slimtsf import (
    SlimTSFClassifier,
    SlidingWindowIntervalTransformer,
    IntervalStatsPoolingTransformer,
)


def generate_synthetic_sensor_data(
    n_samples_per_class: int = 40,
    n_channels: int = 3,
    n_timepoints: int = 120,
    random_state: int = 42,
):
    """
    Generate synthetic tri-axial accelerometer data for 3 activity classes:
      - Class 0 (Resting): Low-amplitude Gaussian noise.
      - Class 1 (Walking): Periodic rhythmic waveform on Channel 0 (X-axis).
      - Class 2 (Running): High-frequency oscillation + step shift on Channels 1 & 2.
    """
    rng = np.random.default_rng(random_state)
    n_classes = 3
    total_samples = n_samples_per_class * n_classes

    X = rng.standard_normal((total_samples, n_channels, n_timepoints)) * 0.5
    y = np.repeat(np.arange(n_classes), n_samples_per_class)

    t = np.linspace(0, 4 * np.pi, n_timepoints)

    # Class 1: Walking rhythm
    idx_c1 = np.where(y == 1)[0]
    for i in idx_c1:
        X[i, 0, :] += np.sin(t) * 2.0
        X[i, 1, :] += np.cos(t) * 1.0

    # Class 2: Running high-frequency acceleration
    idx_c2 = np.where(y == 2)[0]
    for i in idx_c2:
        X[i, 0, :] += np.sin(2.5 * t) * 3.5
        X[i, 1, :] += np.cos(2.5 * t) * 3.0
        X[i, 2, 60:] += 2.0  # abrupt shift in Z-axis

    return X, y


def main():
    print("=" * 70)
    print(" slimtsf: Sliding-Window Multivariate Time-Series Forest")
    print("=" * 70)

    # 1. Generate synthetic 3-channel dataset
    X, y = generate_synthetic_sensor_data(n_samples_per_class=40, random_state=42)
    print(f"Generated dataset: X shape = {X.shape}, y shape = {y.shape}")
    print(f"Classes: {np.unique(y)} (0: Resting, 1: Walking, 2: Running)\n")

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} cases | Test set: {X_test.shape[0]} cases\n")

    # 3. Fit SlimTSFClassifier
    print("Fitting SlimTSFClassifier with bootstrap stability selection...")
    clf = SlimTSFClassifier(
        window_sizes=[16, 32, 64],
        window_step_ratio=0.5,
        feature_functions=["mean", "std", "slope"],
        aggregations=("min", "mean", "max"),
        feature_mode="both",
        bootstrap=True,
        bootstrap_run=10,
        top_rank=5,
        importance_method="gini",
        n_estimators=100,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # 4. Predict and evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Resting", "Walking", "Running"]))

    # 5. Inspect Selected Features
    print("-" * 70)
    print("Top Selected Features across Bootstrap Passes:")
    selection_counts = clf.get_feature_selection_frequencies()
    for rank, (feat_name, count) in enumerate(selection_counts[:5], start=1):
        print(f"  {rank}. {feat_name} (selected in {count}/{clf.bootstrap_run} passes)")

    # 6. Modular Pipeline Demonstration (Stage 1 + Stage 2 only)
    print("-" * 70)
    print("Demonstrating Standalone Feature Transformers:")
    stage1 = SlidingWindowIntervalTransformer(
        window_sizes=[16, 32],
        feature_functions=["mean", "std", "slope"],
    )
    stage2 = IntervalStatsPoolingTransformer(aggregations=("min", "mean", "max"))

    feat_stage1 = stage1.fit_transform(X_train)
    feat_stage2 = stage2.fit_transform(feat_stage1, feature_metadata=stage1.feature_metadata_)

    print(f"  Stage 1 output (interval features): {feat_stage1.shape}")
    print(f"  Stage 2 output (pooled features):   {feat_stage2.shape}")
    print("=" * 70)


if __name__ == "__main__":
    main()
