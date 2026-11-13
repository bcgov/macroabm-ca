"""Reader and container for personal income tax data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PersonalIncomeTaxBracketsData:
    """Container for personal income tax thresholds and rates.

    Holds a DataFrame of personal income tax thresholds and rates for specific years keyed by jurisdiction name.

    Attributes:
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
        reader: PersonalIncomeTaxBracketsReader,
    ) -> PersonalIncomeTaxBracketsData:
        """Build a PersonalIncomeTaxBracketsData container from a reader.

        Args:
            reader: Loaded PersonalIncomeTaxBracketsReader.

        Returns:
            PersonalIncomeTaxBracketsData container ready to be passed to the model.
        """
        return cls(brackets=reader.brackets)


@dataclass
class PersonalIncomeTaxBracketsReader:
    """Reader for a personal income tax brackets CSV.

    The CSV must have years as the row index and jurisdiction, upper, rate as
    column headers.

    Values in the jurisdiction column must match the country codes used by the model
    (e.g. CAN_BC). Values in the upper and threshold columns must comply 
    with PersonalIncomeTax._validate_brackets().

    Attributes:
        brackets: DataFrame loaded from the CSV (None if file absent).
    """

    brackets: Optional[pd.DataFrame] = None

    @classmethod
    def read_from_raw_data(
        cls,
        data_path: Path | str,
    ) -> PersonalIncomeTaxBracketsReader:
        """Load the personal income tax bracket CSV from disk.

        Args:
            data_path: Path to the CSV file (index_col=0 assumed).

        Returns:
            PersonalIncomeTaxBracketsReader with loaded DataFrame (None if file absent).
        """
        if isinstance(data_path, str):
            data_path = Path(data_path)
        brackets = pd.read_csv(data_path, index_col=0) if data_path.exists() else None
        return cls(brackets=brackets)

@dataclass
class PersonalIncomeTaxNonrefundableCreditsData:
    """Container for personal income tax nonrefundable credits data.

    Holds a DataFrame of personal income tax nonrefundable credits for specific years keyed by jurisdiction name.

    Attributes:
        nonrefundable_credits (pd.DataFrame): Nonrefundable credits data
    """

    nonrefundable_credits: Optional[pd.DataFrame] = None

    @property
    def values_dictionary(self) -> dict:
        """Dict-based access to the nonrefundable credits DataFrame."""
        return {"nonrefundable_credits": self.nonrefundable_credits}

    @classmethod
    def from_reader(
        cls,
        reader: PersonalIncomeTaxNonrefundableCreditsReader,
    ) -> PersonalIncomeTaxNonrefundableCreditsData:
        """Build a PersonalIncomeTaxNonrefundableCreditsReader container from a reader.

        Args:
            reader: Loaded PersonalIncomeTaxNonrefundableCreditsReader.

        Returns:
            PersonalIncomeTaxNonrefundableCreditsReader container ready to be passed to the model.
        """
        return cls(nonrefundable_credits=reader.nonrefundable_credits)


@dataclass
class PersonalIncomeTaxNonrefundableCreditsReader:
    """Reader for a personal income tax nonrefundable credits CSV.

    The CSV must have years as the row index and credit, amount, top, rate, clawback, clawback_rate, indexed
    as column headers.

    Values in the jurisdiction column must match the country codes used by the model
    (e.g. CAN_BC). Values in the all other columns must comply 
    with utilization in PersonalIncomeTax.compute_individual_nonrefundable_credits().

    Attributes:
        nonrefundable_credits: DataFrame loaded from the CSV (None if file absent).
    """

    nonrefundable_credits: Optional[pd.DataFrame] = None

    @classmethod
    def read_from_raw_data(
        cls,
        data_path: Path | str,
    ) -> PersonalIncomeTaxNonrefundableCreditsReader:
        """Load the personal income tax nonrefundable credit CSV from disk.

        Args:
            data_path: Path to the CSV file (index_col=0 assumed).

        Returns:
            PersonalIncomeTaxNonrefundableCreditsReader with loaded DataFrame (None if file absent).
        """
        if isinstance(data_path, str):
            data_path = Path(data_path)
        nonrefundable_credits = pd.read_csv(data_path, index_col=0) if data_path.exists() else None
        return cls(nonrefundable_credits=nonrefundable_credits)
