"""Reader and container for personal income tax data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PersonalIncomeTaxData:
    """Container for personal income tax thresholds and rates.

    Holds a DataFrame of personal income tax thresholds and rates for specific years keyed by jurisdiction name.

    Attributes:
        # year: Year where thresholds and rates apply.
        # jurisdiction: Country where thresholds and rates apply.
        brackets (list[tuple[float, float]]): Taxation rate as recorded as (threshold, rate)
    """

    brackets: Optional[pd.DataFrame] = None

    @property
    def values_dictionary(self) -> dict:
        """Dict-based access to the brackets DataFrame."""
        return {"brackets": self.brackets}

    @classmethod
    def from_reader(
        cls,
        reader: PersonalIncomeTaxReader,
    ) -> PersonalIncomeTaxData:
        """Build a PersonalIncomeTaxData container from a reader.

        Args:
            reader: Loaded PersonalIncomeTaxReader.

        Returns:
            PersonalIncomeTaxData container ready to be passed to the model.
        """
        return cls(brackets=reader.brackets)


@dataclass
class PersonalIncomeTaxReader:
    """Reader for a personal income tax brackets CSV.

    The CSV must have years as the row index and jurisdiction, upper, rate as
    column headers.

    Values in the jurisdiction column must match the country codes used by the model
    (e.g. CAN_BC). Values in the upper and threshold columsn must comply 
    with PersonalIncomeTax._validate_brackets().

    Attributes:
        brackets: DataFrame loaded from the CSV (None if file absent).
    """

    brackets: Optional[pd.DataFrame] = None

    @classmethod
    def read_from_raw_data(
        cls,
        data_path: Path | str,
    ) -> PersonalIncomeTaxReader:
        """Load the sector prices CSV from disk.

        Args:
            data_path: Path to the CSV file (index_col=0 assumed).

        Returns:
            PersonalIncomeTaxReader with loaded DataFrame (None if file absent).
        """
        if isinstance(data_path, str):
            data_path = Path(data_path)
        brackets = pd.read_csv(data_path, index_col=0) if data_path.exists() else None
        return cls(brackets=brackets)
