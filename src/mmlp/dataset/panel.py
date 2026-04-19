"""
Yahoo Return Panel
------------------
Build canonical long-format panels from calendar-normalized Yahoo returns for
return forecasting workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from mmlp.dataset.yahoo import YahooDailyReturnsLoader, YahooDailyReturnsRequest

if TYPE_CHECKING:
    from mmlp.config.models import BaseConfigModel as FeatureConfig

__all__ = ["YahooVolatilityPanelBuilder"]


class YahooVolatilityPanelBuilder:
    """
    Build a canonical panel DataFrame for return forecasting.

    Parameters
    ----------
    id_column : str, default="asset_id"
        Identifier column name used in the output panel.
    date_column : str, default="date"
        Date column name used in the output panel.
    return_column : str, default="return"
        Return column name used in the output panel.
    ticker_column : str, default="ticker"
        Ticker column name used as a static categorical feature.
    """

    def __init__(
        self,
        id_column: str = "asset_id",
        date_column: str = "date",
        return_column: str = "return",
        ticker_column: str = "ticker",
        sector_column: str = "sector",
        industry_column: str = "industry",
    ) -> None:
        self.id_column = id_column
        self.date_column = date_column
        self.return_column = return_column
        self.ticker_column = ticker_column
        self.sector_column = sector_column
        self.industry_column = industry_column

    def build_from_loader(
        self,
        *,
        loader: YahooDailyReturnsLoader,
        request: YahooDailyReturnsRequest,
        feature_config: FeatureConfig,
        universe_metadata: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Download returns and build the forecasting panel.

        Parameters
        ----------
        loader : YahooDailyReturnsLoader
            Yahoo Finance loader used to fetch calendar-normalized returns.
        request : YahooDailyReturnsRequest
            Download request validated by the data layer.
        feature_config : FeatureConfig
            Feature configuration defining the forecasting layout.

        Returns
        -------
        pandas.DataFrame
            Canonical long-format panel with transformed returns and ticker.
        """

        returns = loader.load_returns(request=request)
        return self.build_from_returns(
            returns=returns,
            feature_config=feature_config,
            universe_metadata=universe_metadata,
        )

    def build_feature_panel_from_loader(
        self,
        *,
        loader: YahooDailyReturnsLoader,
        request: YahooDailyReturnsRequest,
        universe_metadata: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Download returns and build a raw feature panel without training targets.

        Parameters
        ----------
        loader : YahooDailyReturnsLoader
            Yahoo Finance loader used to fetch calendar-normalized returns.
        request : YahooDailyReturnsRequest
            Download request validated by the data layer.

        Returns
        -------
        pandas.DataFrame
            Canonical long-format feature panel containing raw returns and ticker.
        """

        returns = loader.load_returns(request=request)
        return self.build_feature_panel_from_returns(
            returns=returns,
            universe_metadata=universe_metadata,
        )

    def build_from_returns(
        self,
        *,
        returns: pd.DataFrame,
        feature_config: FeatureConfig,
        universe_metadata: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Convert wide returns into a canonical long-format forecasting panel.

        Parameters
        ----------
        returns : pandas.DataFrame
            Calendar-normalized wide return matrix indexed by date.
        feature_config : FeatureConfig
            Feature configuration defining the return transformation.

        Returns
        -------
        pandas.DataFrame
            Canonical long-format panel with returns and ticker.
        """

        _ = feature_config
        return self.build_feature_panel_from_returns(
            returns=returns,
            universe_metadata=universe_metadata,
        )

    def build_feature_panel_from_returns(
        self,
        *,
        returns: pd.DataFrame,
        universe_metadata: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Convert wide returns into a canonical long-format feature panel.

        Parameters
        ----------
        returns : pandas.DataFrame
            Calendar-normalized wide return matrix indexed by date.

        Returns
        -------
        pandas.DataFrame
            Canonical long-format feature panel with returns and ticker.
        """

        return_frame = returns.stack(future_stack=True).rename(self.return_column).reset_index()
        return_frame.columns = [self.date_column, self.id_column, self.return_column]
        return_frame = self._attach_static_metadata(
            return_frame=return_frame,
            universe_metadata=universe_metadata,
        )
        return return_frame.sort_values([self.id_column, self.date_column]).reset_index(drop=True)

    def transform_feature_panel(
        self,
        *,
        panel: pd.DataFrame,
        feature_config: FeatureConfig,
    ) -> pd.DataFrame:
        """
        Validate and normalize a canonical feature panel without transforming returns.

        Parameters
        ----------
        panel : pandas.DataFrame
            Canonical long-format feature panel containing returns.
        feature_config : FeatureConfig
            Feature configuration defining the forecasting layout.

        Returns
        -------
        pandas.DataFrame
            Canonical long-format panel with standard returns.
        """

        required_columns = {self.date_column, self.id_column, self.return_column}
        missing_columns = sorted(required_columns.difference(panel.columns))
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(f"Feature panel is missing required columns: {missing_text}")
        _ = feature_config
        return panel.sort_values([self.id_column, self.date_column]).reset_index(drop=True)

    def _attach_static_metadata(
        self,
        *,
        return_frame: pd.DataFrame,
        universe_metadata: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """
        Attach static ticker metadata to a long-format return panel.

        Parameters
        ----------
        return_frame : pandas.DataFrame
            Long-format return panel containing the identifier column.
        universe_metadata : pandas.DataFrame or None
            Optional universe metadata keyed by ticker.

        Returns
        -------
        pandas.DataFrame
            Long-format panel with ticker, sector, and industry columns.
        """

        frame = return_frame.copy()
        frame[self.ticker_column] = frame[self.id_column].astype(str)
        if universe_metadata is None:
            frame[self.sector_column] = "unknown"
            frame[self.industry_column] = "unknown"
            return frame

        metadata = universe_metadata.copy()
        required_columns = {self.ticker_column, self.sector_column, self.industry_column}
        missing_columns = sorted(required_columns.difference(metadata.columns))
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(f"Universe metadata is missing required columns: {missing_text}")

        metadata[self.ticker_column] = (
            metadata[self.ticker_column].astype(str).str.strip().str.upper()
        )
        metadata[self.sector_column] = (
            metadata[self.sector_column].astype(str).str.strip().replace("", "unknown")
        )
        metadata[self.industry_column] = (
            metadata[self.industry_column].astype(str).str.strip().replace("", "unknown")
        )
        metadata = metadata.loc[
            :,
            [self.ticker_column, self.sector_column, self.industry_column],
        ].drop_duplicates(
            subset=[self.ticker_column],
            keep="first",
        )

        frame[self.ticker_column] = frame[self.ticker_column].astype(str).str.strip().str.upper()
        frame = frame.merge(metadata, how="left", on=self.ticker_column, validate="many_to_one")
        frame[self.sector_column] = frame[self.sector_column].fillna("unknown")
        frame[self.industry_column] = frame[self.industry_column].fillna("unknown")
        return frame
