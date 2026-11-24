"""Reader and container for consumer data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ConsumerTaxData:
    """Container for consumer tax rate data.

    Holds a DataFrame of consumer tax rate data for specific years keyed by jurisdiction and industry name.

    Attributes:
        data (pd.DataFrame): Consumer tax rate data
    """

    data: Optional[pd.DataFrame] = None

    @property
    def values_dictionary(self) -> dict:
        """Dict-based access to the consumer tax rate DataFrame."""
        return {"data": self.data}

    @classmethod
    def from_reader(
        cls,
        reader: ConsumerTaxReader,
    ) -> ConsumerTaxData:
        """Build a ConsumerTaxData container from a reader.

        Args:
            reader: Loaded ConsumerTaxReader.

        Returns:
            ConsumerTaxData container ready to be passed to the model.
        """
        return cls(data=reader.data)


@dataclass
class ConsumerTaxReader:
    """Reader for a consumer tax rate data CSV.

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
    ) -> ConsumerTaxReader:
        """Load the consumer tax rate CSV from disk.

        Args:
            data_path: Path to the CSV file (index_col=0 assumed).

        Returns:
            ConsumerTaxReader with loaded DataFrame (None if file absent).
        """
        if isinstance(data_path, str):
            data_path = Path(data_path)
        data = pd.read_csv(data_path, index_col=0) if data_path.exists() else None
        return cls(data=data)