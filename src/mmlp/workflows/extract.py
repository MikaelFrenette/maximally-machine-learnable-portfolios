"""
Feature extraction workflow for Yahoo-based raw return panels.
"""

from __future__ import annotations

from pathlib import Path

from mmlp.config.extract import ExtractConfig
from mmlp.config.run import RunConfig
from mmlp.dataset.mace_paper import load_mace_paper_returns
from mmlp.dataset.panel import YahooVolatilityPanelBuilder
from mmlp.dataset.sp500 import load_mace_reference_universe
from mmlp.dataset.yahoo import YahooDailyReturnsLoader, YahooDailyReturnsRequest
from mmlp.preprocessing import build_lagged_feature_panel

__all__ = ["extract_features_from_config", "extract_features_from_run_config"]


def extract_features_from_config(config: ExtractConfig) -> Path:
    """
    Run raw feature extraction from a validated config.

    Parameters
    ----------
    config : ExtractConfig
        Validated extraction configuration.

    Returns
    -------
    pathlib.Path
        Output CSV path written by the workflow.
    """

    loader = YahooDailyReturnsLoader(
        auto_adjust=config.auto_adjust,
        progress=config.progress,
    )
    request = YahooDailyReturnsRequest(
        tickers=config.tickers,
        start_date=config.start_date,
        end_date=config.end_date,
        price_field=config.price_field,
        calendar=config.calendar,
        drop_missing=config.drop_missing,
    )
    panel_builder = YahooVolatilityPanelBuilder()
    panel = panel_builder.build_feature_panel_from_loader(
        loader=loader,
        request=request,
    )

    output_path = config.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)
    return output_path


def extract_features_from_run_config(config: RunConfig) -> Path:
    """
    Run raw feature extraction from the top-level run configuration.

    Parameters
    ----------
    config : RunConfig
        Validated top-level run configuration.

    Returns
    -------
    pathlib.Path
        Output CSV path written by the workflow.
    """

    panel_builder = YahooVolatilityPanelBuilder(
        id_column=config.preprocessing.id_column,
        date_column=config.preprocessing.date_column,
        return_column=config.preprocessing.return_column,
        ticker_column=config.preprocessing.ticker_column,
        sector_column=config.preprocessing.sector_column,
        industry_column=config.preprocessing.industry_column,
    )
    raw_panel = _build_raw_panel_from_run_config(config=config, panel_builder=panel_builder)
    raw_output_path = _resolve_output_path(config=config, filename="panel.csv")
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_panel.to_csv(raw_output_path, index=False)

    feature_panel = build_lagged_feature_panel(
        panel=raw_panel,
        config=config.preprocessing,
    )
    feature_output_path = _resolve_output_path(config=config, filename="features.csv")
    feature_output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_panel.to_csv(feature_output_path, index=False)
    return feature_output_path


def _build_raw_panel_from_run_config(
    config: RunConfig,
    panel_builder: YahooVolatilityPanelBuilder,
):
    if config.dataset.provider == "yahoo":
        tickers = _resolve_universe_tickers(config=config)
        loader = YahooDailyReturnsLoader(
            auto_adjust=config.dataset.auto_adjust,
            progress=config.dataset.progress,
        )
        request = YahooDailyReturnsRequest(
            tickers=tickers,
            start_date=config.dataset.start_date,
            end_date=config.dataset.end_date,
            price_field=config.dataset.price_field,
            calendar=config.dataset.calendar,
            drop_missing=config.dataset.drop_missing,
        )
        return panel_builder.build_feature_panel_from_loader(
            loader=loader,
            request=request,
        )
    if config.dataset.provider == "mace_paper_csv":
        returns = load_mace_paper_returns(
            path=config.dataset.path,
            start_date=config.dataset.start_date,
            end_date=config.dataset.end_date,
            size=config.dataset.universe.size,
        )
        return panel_builder.build_feature_panel_from_returns(returns=returns)
    raise ValueError(f"Unsupported dataset provider: {config.dataset.provider}")


def _resolve_universe_tickers(config: RunConfig) -> tuple[str, ...]:
    """
    Resolve the configured extraction universe into explicit ticker symbols.

    Parameters
    ----------
    config : RunConfig
        Validated top-level run configuration.

    Returns
    -------
    tuple of str
        Explicit ticker universe for the current extraction.
    """

    return load_mace_reference_universe(
        path=config.dataset.universe.path,
        size=config.dataset.universe.size,
    )


def _resolve_output_path(config: RunConfig, filename: str) -> Path:
    """
    Resolve a materialized output path within the configured output directory.

    Parameters
    ----------
    config : RunConfig
        Validated top-level run configuration.
    filename : str
        Standardized artifact filename.

    Returns
    -------
    pathlib.Path
        Destination path for the requested artifact.
    """

    return Path("outputs") / config.run_name / filename
