# TODO

## Foundation

- [ ] Define target package structure for the clean-room rewrite.
- [ ] Choose the canonical Python version and project tooling.
- [ ] Add project metadata and dependency management.
- [ ] Add formatter, linter, test runner, and type-check configuration.

## Reference Mapping

- [ ] Inventory the legacy `MMLP/` workflow from raw inputs to final outputs.
- [ ] Document which legacy behaviors are intentional and which are likely implementation artifacts.
- [ ] Identify parity-critical outputs for regression testing.

## Core Rewrite

- [ ] Implement validated configuration models.
- [ ] Implement data ingestion and dataset schemas.
- [ ] Implement feature engineering pipeline.
- [ ] Implement sampling / bagging utilities.
- [ ] Implement the MACE training loop.
- [ ] Implement prediction APIs.
- [ ] Implement trading and evaluation modules.
- [ ] Optionally port the dormant R-side `build_features` RF augmentation path (for example volatility features appended to MARX inputs) only if a specific experiment requires it.
- [ ] Port the R observation-weight path for the linear step (`obsw`) only if we decide to reproduce bagging/loose-bag variants more closely.

## Verification

- [ ] Add unit tests for helpers, validation, and numerical routines.
- [ ] Add integration tests for train and predict workflows.
- [ ] Add regression tests against locked legacy fixtures.
- [ ] Define acceptance criteria for parity.

## UX and Operations

- [ ] Add CLI or script entrypoints for standard workflows.
- [ ] Add example configs for common runs.
- [ ] Add run metadata and artifact conventions.
- [ ] Write a quickstart for install, train, evaluate, and reproduce.

## Open Questions

- [ ] Decide which legacy behaviors should be preserved exactly.
- [ ] Decide which behaviors should be corrected in the rewrite and flagged as intentional deviations.
- [ ] Decide the initial scope of the first production-quality milestone.
