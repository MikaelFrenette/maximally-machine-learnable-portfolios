from __future__ import annotations

from pathlib import Path

import pytest

from mmlp.config.extract import ExtractConfig, load_extract_config


def test_load_extract_config_validates_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "extract.yaml"
    config_path.write_text(
        "\n".join(
            [
                "tickers: [spy, qqq]",
                "start_date: 2020-01-01",
                "end_date: 2020-02-01",
                "output_path: outputs/features.csv",
                "calendar: xnys",
                "drop_missing: false",
            ]
        ),
        encoding="utf-8",
    )

    config = load_extract_config(config_path)

    assert isinstance(config, ExtractConfig)
    assert config.tickers == ("SPY", "QQQ")
    assert config.calendar == "XNYS"


def test_load_extract_config_rejects_unknown_knob(tmp_path: Path) -> None:
    config_path = tmp_path / "extract.yaml"
    config_path.write_text(
        "\n".join(
            [
                "tickers: [SPY]",
                "start_date: 2020-01-01",
                "end_date: 2020-02-01",
                "output_path: outputs/features.csv",
                "unknown_knob: true",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(Exception):
        load_extract_config(config_path)
