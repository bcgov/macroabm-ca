"""Reader and container for donation data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DonationData:
    """Container for donations data.

    Holds a DataFrame of donations data for specific years keyed by jurisdiction name.

    Attributes:
        data (pd.DataFrame): Donation data for regression model
    """

    data: Optional[pd.DataFrame] = None

    @property
    def values_dictionary(self) -> dict:
        """Dict-based access to the donation DataFrame."""
        return {"data": self.data}

    @classmethod
    def from_reader(
        cls,
        reader: DonationReader,
    ) -> DonationData:
        """Build a DonationData container from a reader.

        Args:
            reader: Loaded DonationReader.

        Returns:
            DonationData container ready to be passed to the model.
        """
        return cls(data=reader.data)


@dataclass
class DonationReader:
    """Reader for a donations data CSV.

    The CSV must have years as the row index and jurisdiction, age, income, and amount as
    column headers.

    Values in the jurisdiction column must match the country codes used by the model
    (e.g. CAN_BC). Values in the age, income, and amount columns must comply 
    with Individuals.update_states_from_data().

    Attributes:
        data: DataFrame loaded from the CSV (None if file absent).
    """

    data: Optional[pd.DataFrame] = None

    @classmethod
    def read_from_raw_data(
        cls,
        data_path: Path | str,
    ) -> DonationReader:
        """Load the donations bracket CSV from disk.

        Args:
            data_path: Path to the CSV file (index_col=0 assumed).

        Returns:
            DonationReader with loaded DataFrame (None if file absent).
        """
        if isinstance(data_path, str):
            data_path = Path(data_path)
        data = pd.read_csv(data_path, index_col=0) if data_path.exists() else None
        return cls(data=data)
