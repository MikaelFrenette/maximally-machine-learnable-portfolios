# AGENT

## Purpose

This repository is for a professional, clean-room rewrite of the MACE model and its surrounding workflow.

The legacy code in `MMLP/` is a behavioral reference only. It exists so the new implementation can be compared against it, not so the old code can become the new system.

## Mission

Build a codebase that is:

- faithful where parity matters
- explicit where the legacy implementation was opaque
- testable end to end
- easy for another engineer to install, run, and extend

## Working Rules

- Do not modify legacy reference code unless explicitly instructed.
- Put all new implementation work into new project structure and modules.
- Keep parity work measurable with fixtures, regression tests, and documented assumptions.
- Prefer typed interfaces, validated config, and deterministic execution.
- Treat documentation, tests, and reproducibility as first-class deliverables.
- Keep responsibilities sharply separated as model code is introduced.
- Default to reusable lower-level components before adding orchestration glue.

## Delivery Standard

Every meaningful implementation change should aim to leave behind:

- code
- tests
- user-facing configuration or entrypoints where needed
- documentation for how to run it

## Expected Rewrite Areas

- data loading and dataset contracts
- feature engineering
- MACE training loop
- bagging / sampling logic
- prediction interfaces
- trading and evaluation workflow
- experiment configuration and run metadata

## Modularization Expectations

As core model code is added, the repository should converge toward clear layers:

- contracts and schemas
- dataset and panel builders
- preprocessing transforms
- estimator internals
- fitted model objects
- workflows and CLI entrypoints

The intended direction is:

- workflows compose components
- components expose stable interfaces
- numerical logic remains reusable outside a specific workflow

## Implementation Bias

When there is a choice, prefer:

- explicit contracts over implicit conventions
- small reusable functions over long stateful methods
- typed fitted-model objects over ad hoc result dictionaries
- isolated numerical components over workflow-coupled logic
- separate training and inference paths where their responsibilities differ

Avoid building a second monolith in cleaner syntax.

## Interaction Policy

When working in this repository:

- surface correctness risks early
- challenge legacy behavior when it appears accidental rather than intentional
- preserve a clear distinction between `paper-faithful` and `improved` behavior
- avoid speculative abstractions before the core path is working

## Definition of Done

A component is not done when it merely runs once. It is done when:

- its contract is clear
- its behavior is test-covered
- its configuration is validated
- its usage is documented
- its outputs are reproducible
