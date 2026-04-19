"""
Generate plots from previously materialized run artifacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmlp.config.run import load_run_config
from mmlp.workflows import generate_plots_from_config


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser for plot generation.
    """

    parser = argparse.ArgumentParser(description="Generate plots for an existing MMLP run.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the top-level run YAML configuration file.",
    )
    return parser


def main() -> None:
    """
    Generate plots from parsed command-line arguments.
    """

    args = build_argument_parser().parse_args()
    config = load_run_config(args.config)
    artifacts = generate_plots_from_config(config)
    print(f"cumulative_plot_path={artifacts.cumulative_plot_path}")
    print(f"yearly_heatmap_path={artifacts.yearly_heatmap_path}")


if __name__ == "__main__":
    main()
