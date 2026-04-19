# Cleanup

This file tracks code that appears dead, near-dead, or outside the active project path.

Nothing listed here should be deleted automatically. The purpose is to document cleanup candidates before removal or refactor.

## Currently Dead or Near-Dead

### Generic Alternating Prototype Path

These modules belong to the earlier generic alternating-regression scaffold rather than the active MACE-specific implementation:

- [`src/mmlp/model/alternating.py`](/mnt/c/users/user/dropbox/mace/src/mmlp/model/alternating.py)
- [`src/mmlp/model/fitted.py`](/mnt/c/users/user/dropbox/mace/src/mmlp/model/fitted.py)
- [`src/mmlp/model/input.py`](/mnt/c/users/user/dropbox/mace/src/mmlp/model/input.py)
- [`src/mmlp/model/schema.py`](/mnt/c/users/user/dropbox/mace/src/mmlp/model/schema.py)

Why they are listed:

- the active run workflow uses [`src/mmlp/model/mace.py`](/mnt/c/users/user/dropbox/mace/src/mmlp/model/mace.py)
- the current pipeline does not call the generic alternating estimator
- these modules are still exported and tested, but they are not part of the main MACE run path

Recommended decision later:

- either remove them
- or explicitly reclassify them as an archival / baseline model path

### Extraction-Only Config / CLI Path

These pieces are still functional, but they are no longer the primary public interface:

- [`src/mmlp/config/extract.py`](/mnt/c/users/user/dropbox/mace/src/mmlp/config/extract.py)
- [`src/mmlp/workflows/extract.py`](/mnt/c/users/user/dropbox/mace/src/mmlp/workflows/extract.py): `extract_features_from_config(...)`
- [`scripts/extract_features.py`](/mnt/c/users/user/dropbox/mace/scripts/extract_features.py)
- [`configs/extract.example.yaml`](/mnt/c/users/user/dropbox/mace/configs/extract.example.yaml)

Why they are listed:

- the project now centers around the top-level run config and [`scripts/run_pipeline.py`](/mnt/c/users/user/dropbox/mace/scripts/run_pipeline.py)
- the extraction-only path is a sidecar interface rather than the canonical workflow

Recommended decision later:

- either keep them as explicit utility tooling
- or remove them and standardize everything on the run config

### Generated Metadata / Cache Files

These are not source code and should not be treated as part of the implementation:

- [`src/mmlp.egg-info/`](/mnt/c/users/user/dropbox/mace/src/mmlp.egg-info)
- [`src/mmlp/__pycache__/`](/mnt/c/users/user/dropbox/mace/src/mmlp/__pycache__)

Recommended decision later:

- keep them out of version control if they are currently tracked
- remove them in a cleanup commit when appropriate

## Not Cleanup Targets

These are active parts of the current project and should not be treated as dead code:

- [`src/mmlp/model/mace.py`](/mnt/c/users/user/dropbox/mace/src/mmlp/model/mace.py)
- [`src/mmlp/workflows/run.py`](/mnt/c/users/user/dropbox/mace/src/mmlp/workflows/run.py)
- [`src/mmlp/workflows/plotting.py`](/mnt/c/users/user/dropbox/mace/src/mmlp/workflows/plotting.py)
- [`src/mmlp/evaluation/`](/mnt/c/users/user/dropbox/mace/src/mmlp/evaluation)
- [`src/mmlp/plotting/`](/mnt/c/users/user/dropbox/mace/src/mmlp/plotting)
- [`scripts/run_pipeline.py`](/mnt/c/users/user/dropbox/mace/scripts/run_pipeline.py)
- [`scripts/plot_results.py`](/mnt/c/users/user/dropbox/mace/scripts/plot_results.py)
- [`run_demo.sh`](/mnt/c/users/user/dropbox/mace/run_demo.sh)

## Guiding Rule

Cleanup should follow this order:

1. confirm the code is not on the active path
2. document the rationale here
3. remove the code only in a dedicated cleanup change
