"""
Feature Extraction CLI
----------------------
Command-line entrypoint for Yahoo-based raw feature extraction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmlp.config.run import load_run_config
from mmlp.workflows.extract import extract_features_from_run_config


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser for feature extraction runs.

    Returns
    -------
    argparse.ArgumentParser
        Configured extraction CLI parser.
    """

    parser = argparse.ArgumentParser(
        description="Extract Yahoo-based raw feature panels to CSV."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the top-level run YAML configuration file.",
    )
    return parser


def main() -> None:
    """
    Run the extraction CLI from parsed command-line arguments.

    Returns
    -------
    None
        This function executes the extraction and preprocessing workflow and
        prints the feature-panel output path.
    """

    args = build_argument_parser().parse_args()
    config = load_run_config(args.config)
    output_path = extract_features_from_run_config(config)
    print(f"features_path={output_path}")


if __name__ == "__main__":
    main()
