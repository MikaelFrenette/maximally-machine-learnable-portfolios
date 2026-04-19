# Conventions

This repository is for a clean-room rewrite of the MACE model and its supporting workflow.

The legacy implementation under `MMLP/` is reference material only. Do not patch, refactor, or extend legacy code unless the task explicitly says to do so.

## Development Behavior

- Build new code in a separate, modern project structure.
- Treat the rewrite as production-oriented research software, not as an exploratory notebook dump.
- Prefer small, reviewable changes over broad rewrites without checkpoints.
- Preserve a clear boundary between `reference implementation` and `new implementation`.
- Favor explicitness over cleverness.
- Do not add convenience behavior that hides errors or silently changes user intent.

## Primary Goals

- Reproduce the original MACE behavior where intended.
- Make the new implementation easy to install, configure, test, and run.
- Keep the codebase understandable to a new engineer without oral context.
- Design for repeatable experiments and traceable outputs.

## Project Structure

New code should be organized by responsibility.

- `src/`: library code only
- `tests/`: unit, integration, and regression tests
- `configs/`: user-facing configuration files
- `scripts/`: thin entrypoints and operational utilities
- `docs/`: design notes, user guides, and reproducibility docs
- `artifacts/` or `outputs/`: generated results, never imported as source

Rules:

- Keep modules focused on one responsibility.
- Avoid large god-files.
- Do not place core logic in notebooks.
- Notebooks, if used, are for exploration only and must not be the system of record.
- Public CLI behavior should live in scripts that call library code, not duplicate it.

## Python Standards

- Target Python 3.11+ unless the repo states otherwise.
- Use type hints on all public functions, methods, and classes.
- Prefer `pathlib.Path` over manual path string handling.
- Prefer dataclasses or `pydantic` models for structured data over loose dictionaries.
- Prefer explicit imports. Do not use wildcard imports.
- Avoid mutable global state.
- Avoid hidden side effects during import.

## API and Design Rules

- Separate pure computation from I/O, persistence, plotting, and CLI concerns.
- Keep configuration parsing separate from model logic.
- Keep feature engineering separate from model fitting.
- Keep training, prediction, evaluation, and trading logic in distinct modules.
- Design public interfaces around typed objects, not ad hoc nested dictionaries.
- Make randomness explicit and seedable.

## Responsibility Boundaries

As model implementation begins, each layer must have a narrowly defined role.

- `config/` defines validated user-facing schemas only.
- `dataset/` handles data access, panel construction, and dataset contracts.
- `preprocessing/` transforms validated panels into model-ready features.
- `model/` contains model schemas, numerical routines, estimators, and inference code.
- `workflows/` orchestrates multi-step runs by composing lower-level modules.
- `scripts/` are thin entrypoints only and must not contain core business logic.

Rules:

- A module should own one reason to change.
- Prefer composition over large classes with mixed responsibilities.
- Keep numerical kernels independent from CLI, file paths, and logging.
- Keep orchestration code separate from reusable model primitives.
- If logic can be reused across workflows, it belongs in `src/mmlp/`, not in `scripts/`.

## Model Implementation Rules

- Define clear input and output schemas before writing training code.
- Separate model input validation from model fitting.
- Separate feature selection from estimator logic.
- Separate training-time state from inference-time state.
- Keep serialization concerns separate from estimator mathematics.
- Prefer small reusable functions for math-heavy transformations.
- Encapsulate learned parameters in explicit typed objects rather than loose dictionaries.
- Make intermediate representations explicit when they are passed across stages.

When implementing MACE-specific components, prefer a structure like:

- schema / contracts
- feature matrix assembly
- sampling / bagging utilities
- estimator core
- fitted model object
- prediction utilities
- evaluation adapters

## Reuse Rules

- Reuse should come from clean interfaces, not from shared mutable state.
- Do not duplicate the same transformation in dataset, preprocessing, and model layers.
- If two modules need the same helper, promote it into a dedicated reusable component.
- Avoid convenience wrappers that hide too much state or blur ownership.
- Reuse is good only if the resulting abstraction is still easier to understand than the duplicated code.

## Validation

Use `pydantic` for user-facing configuration and external inputs.

Rules:

- Fail fast on invalid input.
- Do not silently coerce semantically invalid values.
- Do not invent fallback defaults outside the declared schema.
- Raise clear validation errors with enough context to fix the input.

## Documentation

Every public module must start with a short top-of-file docstring explaining its responsibility.

Every public class, function, and method should have a concise numpydoc-style docstring when the interface is non-trivial.

Minimum expectations:

- what the component does
- key parameters
- return value
- important invariants or failure modes

Documentation is part of the implementation, not cleanup.

## Testing

Testing is required for all non-trivial behavior.

Expected test layers:

- unit tests for math, transforms, and validation
- integration tests for training and prediction flows
- regression tests for parity-critical behavior against locked fixtures

Rules:

- Add tests with the code change, not later.
- Prefer deterministic tests.
- Seed stochastic components explicitly.
- Do not rely on external services in default test runs.
- Keep test fixtures small enough for fast iteration.

## Reproducibility

- Training runs must be reproducible from code, config, and seed.
- Save enough metadata to identify the exact code path and configuration used.
- Avoid hidden environment-dependent behavior.
- Make defaults explicit in config or code, not implicit in scattered scripts.

## Configuration

Public config files should be readable by a user without opening the source code.

Rules:

- Comment meaningful knobs.
- If a config field is effectively an enum / literal, list the allowed values inline in the YAML
  comment next to that field.
- Keep field names stable and descriptive.
- Group related settings together.
- Avoid ambiguous abbreviations in public config.
- Keep config schema versionable if the surface area grows.

## Logging and Errors

- Use structured, informative logging for long-running workflows.
- Raise errors instead of printing and continuing when invariants break.
- Error messages should identify the failing input, stage, or component.
- Avoid noisy logging in library code; reserve progress reporting for scripts and apps.

## Data and Artifacts

- Never overwrite source data silently.
- Treat raw data, processed data, and generated artifacts as separate concerns.
- Record how processed datasets were created.
- Keep large or generated outputs out of importable source directories.

## Review Standards

Code is not ready if any of the following are true:

- the behavior is correct but the interface is confusing
- the code cannot be tested in isolation
- the change mixes unrelated concerns
- the implementation depends on undocumented assumptions
- failure modes are hidden
- the ownership boundary of the module is unclear
- reusable logic is trapped inside a workflow or script

## Prohibited Patterns

- editing legacy `MMLP/` code as a substitute for writing the new implementation
- shipping notebook-only workflows
- relying on unnamed tuple/list positions for public interfaces
- passing nested untyped "backpack" dictionaries through the whole system
- using magic constants without naming or documenting them
- mixing training logic with plotting/export side effects
- adding broad fallback behavior to mask invalid inputs
- putting reusable model logic directly inside CLI scripts
- combining schema validation, feature generation, fitting, and serialization in one file
- using one class as config holder, trainer, predictor, and artifact writer at the same time

## Default Tooling Expectations

Unless the repo later standardizes something different:

- formatter: `ruff format`
- linting: `ruff check`
- tests: `pytest`
- type checking: `pyright` or `mypy`

Tooling may evolve, but consistency is mandatory once chosen.
