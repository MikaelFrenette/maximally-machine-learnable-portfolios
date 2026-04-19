"""
Run the end-to-end MMLP pipeline from a validated run configuration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmlp.config.run import load_run_config
from mmlp.workflows import run_pipeline_from_config


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser for end-to-end runs.
    """

    parser = argparse.ArgumentParser(description="Run the end-to-end MMLP pipeline.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the top-level run YAML configuration file.",
    )
    return parser


def main() -> None:
    """
    Run the pipeline from parsed command-line arguments.
    """

    args = build_argument_parser().parse_args()
    config = load_run_config(args.config)
    artifacts = run_pipeline_from_config(config)
    print(f"raw_panel_path={artifacts.raw_panel_path}")
    print(f"feature_panel_path={artifacts.feature_panel_path}")
    print(f"diagnostics_path={artifacts.diagnostics_path}")
    print(f"predictions_path={artifacts.predictions_path}")
    print(f"weights_raw_path={artifacts.weights_raw_path}")
    print(f"weights_normalized_path={artifacts.weights_normalized_path}")
    print(f"summary_path={artifacts.summary_path}")
    print(f"trading_path={artifacts.trading_path}")
    print(f"trading_summary_path={artifacts.trading_summary_path}")
    print(f"trading_yearly_summary_path={artifacts.trading_yearly_summary_path}")
    print(f"cumulative_plot_path={artifacts.cumulative_plot_path}")
    print(f"yearly_heatmap_path={artifacts.yearly_heatmap_path}")
    print(f"log_path={artifacts.log_path}")


if __name__ == "__main__":
    main()
