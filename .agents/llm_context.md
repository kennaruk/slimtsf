# SlimTSF Architecture & Context

SlimTSF (Sliding-Window Multivariate Time-Series Forest) is a scikit-learn compatible estimator for timeseries classification. 
The system avoids traditional distance-based methods or deep learning, relying instead on a highly parallelizable feature-engineering pipeline followed by Random Forest classification.

## Core Stages

1. **Stage 1: `SlidingWindowIntervalTransformer`**
   - Slices input 3-D series `(cases, channels, timepoints)` into sliding windows of various scales.
   - Extracts localized features (mean, standard deviation, slope, custom lambdas) for each window.
2. **Stage 2: `IntervalStatsPoolingTransformer`**
   - Given the localized windows, it pools statistics (min, mean, max) across the temporal axis for each logical `(channel, feature)` group. 
   - Yields a highly compressed 2-D representation of global patterns.
   - *Note: `feature_mode` (`"interval"`, `"pooled"`, `"both"`) decides which outputs traverse to Stage 3.*
3. **Stage 3: Bootstrap Feature Selection (Optional)**
   - Fits an ensemble of shallow Random Forests over the 2-D features.
   - Evaluates feature importance (using Gini, Permutation, SHAP, Fisher, or ANOVA-F) and selects the Top `log2(N)` most consistently important features across passes to reduce dimensionality.
4. **Stage 4: `RandomForestClassifier`**
   - The final scikit-learn classifier fit on the distilled, low-noise feature matrix.

## Development Constraints
- Use standard `pytest` testing for modifications (see `slimtsf_tdd_workflow` skill).
- Maintain parity with `scikit-learn` estimator API.
- Maintain minimal dependencies to preserve library nimbleness.
