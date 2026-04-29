# CHANGELOG

<!-- version list -->

## v1.5.0 (2026-04-29)

### Features

- Standardise bootstrap feature selection across all methods
  ([`f992da4`](https://github.com/kennaruk/slimtsf/commit/f992da4987b1645de253bde77ec63d1547b3a609))


## v1.4.1 (2026-04-24)

### Bug Fixes

- **get_feature_selection_frequencies**: Fix logic supporting
  ([`bed0deb`](https://github.com/kennaruk/slimtsf/commit/bed0deba0dcfe3d9d4547a2096507a0bab0b7295))


## v1.4.0 (2026-04-24)

### Chores

- Add tests for multi cores
  ([`ffe6228`](https://github.com/kennaruk/slimtsf/commit/ffe6228a0c4ef1400c1b9e6ec582a44d90348b6d))

### Features

- **get_feature_names_out**: Add `get_feature_names_out` function
  ([`cf4a4f6`](https://github.com/kennaruk/slimtsf/commit/cf4a4f680ec203e86f7f59aa6c9fc0a81422257b))


## v1.3.3 (2026-04-21)

### Bug Fixes

- **Closure bottleneck**: Improve performance of multi cores
  ([`cd560b6`](https://github.com/kennaruk/slimtsf/commit/cd560b680b3f5c368f1eac7dd079651d8191535c))


## v1.3.2 (2026-04-20)

### Bug Fixes

- **Bottleneck**: Enhance bottleneck of CPU utilization at bootstrapping step, add `verbose` support
  for debugging purpose
  ([`a0a6ea6`](https://github.com/kennaruk/slimtsf/commit/a0a6ea6804e26191dd9161c85548d01576aa0bcd))


## v1.3.1 (2026-04-16)

### Bug Fixes

- Optimize bootstrap logic
  ([`1c81f39`](https://github.com/kennaruk/slimtsf/commit/1c81f399a3dbd331427dc72c9c166f759509d86d))


## v1.3.0 (2026-04-16)

### Features

- **feature_mode / importance_method**: Support feature_modes and new fisher/anova-f feature
  importance
  ([`bd4c2e6`](https://github.com/kennaruk/slimtsf/commit/bd4c2e63471f3eaf12bc28fd281eb275ac645ebe))


## v1.2.1 (2026-03-25)

### Bug Fixes

- Second transformer should concat result with first transformer
  ([`a70ebb6`](https://github.com/kennaruk/slimtsf/commit/a70ebb6366fbf18ea35f9b582995c1e6180122ce))


## v1.2.0 (2026-03-21)

### Features

- Supports aggregations=None to skip, adds permutation and SHAP as alternate feature importance
  functions
  ([`9966397`](https://github.com/kennaruk/slimtsf/commit/996639741018248643379d69757bcc145bfe8439))


## v1.1.0 (2026-03-14)

### Features

- Add pycatch22, antropy, and basic statistical feature functions
  ([`9fed0cb`](https://github.com/kennaruk/slimtsf/commit/9fed0cb631e86fdbc709eaefed743f903b79bf9c))


## v1.0.3 (2026-03-14)

### Bug Fixes

- **bootstrap**: Fix bootstrap logic
  ([`b649770`](https://github.com/kennaruk/slimtsf/commit/b6497708711278d8d53f4385560b3f2598bd1403))


## v1.0.2 (2026-03-09)

### Bug Fixes

- Ci/cd twine publish
  ([`0c05fac`](https://github.com/kennaruk/slimtsf/commit/0c05fac3f08061d7be98d3e7d6f9a37ef733061a))


## v1.0.0 (2026-03-09)

- Initial Release

## v1.0.1 (2026-03-09)

### Bug Fixes

- **CI/CD**: Correct env pypi key
  ([`fdbd0be`](https://github.com/kennaruk/slimtsf/commit/fdbd0be9c6987c320373867368e71b8f6e792eb2))


## v1.0.0 (2026-03-09)

- Initial Release
