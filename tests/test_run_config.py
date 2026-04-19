from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmlp.config.run import RunConfig, load_run_config


def test_load_run_config_validates_nested_sections(tmp_path: Path) -> None:
    universe_path = tmp_path / "mace_universe.json"
    universe_path.write_text(
        json.dumps({"equities": ["AAPL", "MSFT"], "bonds": [], "contracts": [], "forex": []}),
        encoding="utf-8",
    )
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run_name: demo_run",
                "dataset:",
                "  provider: yahoo",
                "  calendar: XNYS",
                '  price_field: "Adj Close"',
                "  start_date: 2020-01-01",
                "  end_date: 2020-02-01",
                "  drop_missing: false",
                "  auto_adjust: false",
                "  progress: false",
                "  universe:",
                f"    path: {universe_path}",
                "    size: 20",
                "split:",
                "  train_start: 2020-01-01",
                "  train_end: 2020-01-15",
                "  test_start: 2020-01-16",
                "  test_end: 2020-02-01",
                "outputs:",
                "  log_level: INFO",
                "  verbosity: 1",
            ]
        ),
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert isinstance(config, RunConfig)
    assert config.dataset.universe.path == universe_path
    assert config.dataset.universe.size == 20
    assert config.split.train_end.isoformat() == "2020-01-15"
    assert config.preprocessing.lags == 5
    assert config.model.device == "cpu"
    assert config.outputs.verbosity == 1


def test_load_run_config_rejects_non_json_universe_path(tmp_path: Path) -> None:
    universe_path = tmp_path / "universe.txt"
    universe_path.write_text("not json", encoding="utf-8")
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run_name: demo_run",
                "dataset:",
                "  provider: yahoo",
                "  calendar: XNYS",
                '  price_field: "Adj Close"',
                "  start_date: 2020-01-01",
                "  end_date: 2020-02-01",
                "  universe:",
                f"    path: {universe_path}",
                "split:",
                "  train_start: 2020-01-01",
                "  train_end: 2020-01-15",
                "  test_start: 2020-01-16",
                "  test_end: 2020-02-01",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(Exception):
        load_run_config(config_path)


def test_load_run_config_accepts_local_mace_paper_csv_without_universe_path(tmp_path: Path) -> None:
    dataset_path = tmp_path / "daily__MACE_paper.csv"
    dataset_path.write_text("date,AAPL\n2020-01-01,0.01\n", encoding="utf-8")
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run_name: demo_run",
                "dataset:",
                "  provider: mace_paper_csv",
                f"  path: {dataset_path}",
                "  calendar: XNYS",
                '  price_field: "Adj Close"',
                "  start_date: 2020-01-01",
                "  end_date: 2020-02-01",
                "  universe:",
                "    size: 20",
                "split:",
                "  train_start: 2020-01-01",
                "  train_end: 2020-01-15",
                "  test_start: 2020-01-16",
                "  test_end: 2020-02-01",
                "outputs:",
                "  log_level: INFO",
                "  verbosity: 1",
            ]
        ),
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.dataset.provider == "mace_paper_csv"
    assert config.dataset.path == dataset_path
    assert config.dataset.universe.size == 20
