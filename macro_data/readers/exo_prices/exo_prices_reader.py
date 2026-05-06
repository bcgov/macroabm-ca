"""Reader and container for exogenous firm price data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class FirmExoPrices:
    """Container for exogenous firm price paths.

    Holds a DataFrame of price trajectories keyed by industry name.
    Each column is an industry; the index is years. Prices are normalised
    to initial_year at runtime by FirmExogenousPriceSetter.

    Attributes:
        prices: DataFrame with years as index and industry names as columns.
        initial_year: Base year for price normalisation.
        initial_model_prices: Per-firm base prices injected from the model
            before the price setter is used.
    """

    prices: Optional[pd.DataFrame] = None
    initial_year: int = 2014
    initial_model_prices: Optional[np.ndarray] = None

    @property
    def values_dictionary(self) -> dict:
        """Dict-based access to the price DataFrame."""
        return {"prices": self.prices}

    @classmethod
    def from_reader(
        cls,
        reader: FirmExoPricesReader,
        initial_year: int = 2014,
    ) -> FirmExoPrices:
        """Build a FirmExoPrices container from a reader.

        Args:
            reader: Loaded FirmExoPricesReader.
            initial_year: Base year for price normalisation.

        Returns:
            FirmExoPrices container ready to be passed to Firms.
        """
        return cls(prices=reader.prices, initial_year=initial_year)


@dataclass
class FirmExoPricesReader:
    """Reader for a single exogenous firm prices CSV.

    The CSV must have years as the row index and industry codes as column
    headers. Column names must match the ISIC Rev. 4 codes used by the model —
    either aggregated (e.g. "D", "B", "C19") or detailed (e.g. "B05", "C19",
    "D") depending on the industries list the model is configured with.
    Values are prices in any consistent unit; the price setter normalises them
    to the initial year automatically.

    Example CSV layout (aggregated industries, initial_year=2014):

        year,D,B
        2013,100.0,80.0
        2014,102.5,85.0
        2015,106.0,91.0
        2030,130.0,110.0

    Start one year before initial_year so that Q1 of the first simulation
    year (which maps to initial_year − 0.25) stays within the interpolation
    range.

    Attributes:
        prices: DataFrame loaded from the CSV (None if file absent).
    """

    prices: Optional[pd.DataFrame] = None

    @classmethod
    def read_from_raw_data(
        cls,
        prices_path: Path | str,
    ) -> FirmExoPricesReader:
        """Load the firm prices CSV from disk.

        Args:
            prices_path: Path to the CSV file (index_col=0 assumed).

        Returns:
            FirmExoPricesReader with loaded DataFrame (None if file absent).
        """
        if isinstance(prices_path, str):
            prices_path = Path(prices_path)
        prices = pd.read_csv(prices_path, index_col=0) if prices_path.exists() else None
        return cls(prices=prices)
