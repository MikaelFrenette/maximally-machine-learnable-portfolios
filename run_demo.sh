#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${ROOT_DIR}/configs/demo_run.yaml"

python3 "${ROOT_DIR}/scripts/run_pipeline.py" --config "${CONFIG_PATH}"
python3 "${ROOT_DIR}/scripts/plot_results.py" --config "${CONFIG_PATH}"
