"""Dividend tax credit schedule — Canadian gross-up and provincial DTC rates.

This module reads a CSV of dividend gross-up and dividend-tax-credit (DTC)
rates and supplies, for a given tax year, the rates that apply to each
dividend type (``eligible`` / ``non_eligible``).

These are published statutory rates that change across years, so they form a
*schedule* (a multi-row, year-ranged table) rather than a scalar modelling
assumption.  They are therefore kept in a CSV in the taxation directory
(``raw_data_path / "taxation" / "personal_income_tax"``, with
``spoof_data/freda/personal_income_tax`` as the committed
fallback) alongside the PIT bracket and tax-credit schedules, not in
``tax_parameters.yaml`` (which holds scalar assumptions only).

CSV format
----------
::

    dividend_type,year_from,year_to,gross_up_rate,bc_dtc_pct_of_grossed_up,bc_dtc_pct_of_actual,notes
    eligible,2012,2018,0.38,0.10,0.138,2014 base-year row
    eligible,2019,,0.38,0.12,0.1656,
    non_eligible,2014,2015,0.18,0.0259,0.0306,2014 base-year row
    ...

Columns:
    - ``dividend_type`` (str): ``eligible`` or ``non_eligible``.
    - ``year_from`` (int): First tax year the row applies to (inclusive).
    - ``year_to`` (int, optional): Last tax year the row applies to
      (inclusive).  Empty means open-ended — the row applies to
      ``year_from`` and every later year.
    - ``gross_up_rate`` (float): Gross-up rate; the taxable dividend is
      ``(1 + gross_up_rate) x cash``.
    - ``bc_dtc_pct_of_grossed_up`` (float): Provincial DTC as a fraction
      of the grossed-up dividend.  This is the rate the model applies.
    - ``bc_dtc_pct_of_actual`` (float, optional): Provincial DTC as a
      fraction of the actual (cash) dividend.  Recorded for reference.
    - ``notes`` (str, optional): Free-text provenance.  Not read at runtime.

For a given tax year, exactly one row must apply per dividend type; a missing
or overlapping range raises ``ValueError``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# ── required CSV columns ──────────────────────────────────────────────
_DTC_REQUIRED_COLS = {
    "dividend_type",              # str — "eligible" / "non_eligible"
    "year_from",                  # int — first applicable year (inclusive)
    "gross_up_rate",              # float — taxable = (1 + rate) x cash
    "bc_dtc_pct_of_grossed_up",   # float — DTC as a fraction of grossed-up
}

# ── recognised dividend types ─────────────────────────────────────────
_DIVIDEND_TYPES = ("eligible", "non_eligible")


@dataclass(frozen=True)
class DividendRates:
    """Gross-up and DTC rates for one dividend type in one tax year.

    Attributes:
        dividend_type: ``"eligible"`` or ``"non_eligible"``.
        gross_up_rate: Gross-up rate; taxable dividend = ``(1 + rate) x cash``.
        dtc_rate_of_grossed_up: Provincial DTC as a fraction of the
            grossed-up dividend (the rate the model applies).
        dtc_rate_of_actual: Provincial DTC as a fraction of the actual cash
            dividend, when recorded.  ``None`` when absent from the CSV.
    """

    dividend_type: str
    gross_up_rate: float
    dtc_rate_of_grossed_up: float
    dtc_rate_of_actual: Optional[float] = None


class DividendTaxCreditSchedule:
    """Year-ranged dividend gross-up and DTC rates, keyed by dividend type.

    Typical usage::

        schedule = DividendTaxCreditSchedule.from_name(
            "bc_dividend_tax_credit_schedule.csv"
        )
        rates = schedule.get_year_rates(tax_year=2014)
        rates["eligible"].gross_up_rate          # -> 0.38
        rates["non_eligible"].dtc_rate_of_grossed_up  # -> 0.0259
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.copy()

    # ── factories ─────────────────────────────────────────────────────

    @classmethod
    def from_csv(cls, path: str | Path) -> "DividendTaxCreditSchedule":
        """Load the dividend-rate schedule from a CSV file.

        Args:
            path: Path to the CSV (e.g. ``bc_dividend_tax_credit_schedule.csv``).

        Returns:
            A configured ``DividendTaxCreditSchedule``.

        Raises:
            ValueError: If required columns are missing.
        """
        df = pd.read_csv(Path(path))

        # Normalise column names (lower-case, spaces to underscores).
        df = df.rename(columns={c: c.lower().replace(" ", "_") for c in df.columns})

        missing = _DTC_REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"Dividend-rate CSV is missing required columns: {sorted(missing)}. "
                f"Found: {sorted(df.columns)}"
            )

        df["dividend_type"] = df["dividend_type"].astype(str).str.strip().str.lower()
        df["year_from"] = df["year_from"].astype(int)
        # year_to may be blank (open-ended); keep as nullable float.
        if "year_to" in df.columns:
            df["year_to"] = pd.to_numeric(df["year_to"], errors="coerce")
        else:
            df["year_to"] = float("nan")
        for col in ("gross_up_rate", "bc_dtc_pct_of_grossed_up"):
            df[col] = df[col].astype(float)
        if "bc_dtc_pct_of_actual" in df.columns:
            df["bc_dtc_pct_of_actual"] = pd.to_numeric(
                df["bc_dtc_pct_of_actual"], errors="coerce"
            )

        return cls(df)

    @classmethod
    def from_name(
        cls,
        filename: str,
        schedule_dir: Path,
    ) -> "DividendTaxCreditSchedule":
        """Load the schedule by filename from *schedule_dir*.

        Args:
            filename: CSV filename (e.g. ``"bc_dividend_tax_credit_schedule.csv"``).
            schedule_dir: Directory holding the schedule CSVs — typically
                ``raw_data_path / "taxation" / "personal_income_tax"``.

        Returns:
            A configured ``DividendTaxCreditSchedule``.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        directory = Path(schedule_dir)
        path = directory / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Dividend-rate file not found: {path}\n"
                f"Available: {sorted(p.name for p in directory.glob('*.csv'))}"
            )
        return cls.from_csv(path)

    # ── public methods ────────────────────────────────────────────────

    def get_rates(self, tax_year: int, dividend_type: str) -> DividendRates:
        """Return the rates applying to *dividend_type* in *tax_year*.

        Args:
            tax_year: The tax year to look up.
            dividend_type: ``"eligible"`` or ``"non_eligible"``.

        Returns:
            The matching :class:`DividendRates`.

        Raises:
            ValueError: If no row, or more than one row, applies.
        """
        dtype = str(dividend_type).strip().lower()
        sub = self._df[self._df["dividend_type"] == dtype]
        if sub.empty:
            raise ValueError(
                f"No rows for dividend_type '{dtype}'. "
                f"Available: {sorted(self._df['dividend_type'].unique())}"
            )

        year_to = sub["year_to"]
        applies = (sub["year_from"] <= tax_year) & (
            year_to.isna() | (tax_year <= year_to)
        )
        matches = sub[applies]

        if matches.empty:
            raise ValueError(
                f"No '{dtype}' dividend rate row covers tax year {tax_year}. "
                f"Covered ranges: "
                f"{[(int(a), None if pd.isna(b) else int(b)) for a, b in zip(sub['year_from'], sub['year_to'])]}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Overlapping '{dtype}' dividend rate rows for tax year {tax_year}: "
                f"{len(matches)} rows match (ranges must be disjoint)."
            )

        row = matches.iloc[0]
        actual = row.get("bc_dtc_pct_of_actual")
        return DividendRates(
            dividend_type=dtype,
            gross_up_rate=float(row["gross_up_rate"]),
            dtc_rate_of_grossed_up=float(row["bc_dtc_pct_of_grossed_up"]),
            dtc_rate_of_actual=(
                None if actual is None or pd.isna(actual) else float(actual)
            ),
        )

    def get_year_rates(self, tax_year: int) -> dict[str, DividendRates]:
        """Return the rates for every dividend type in *tax_year*.

        Args:
            tax_year: The tax year to look up.

        Returns:
            Dict mapping each dividend type (``"eligible"`` /
            ``"non_eligible"``) to its :class:`DividendRates`.

        Raises:
            ValueError: If any dividend type has no applicable row.
        """
        return {
            dtype: self.get_rates(tax_year, dtype) for dtype in _DIVIDEND_TYPES
        }
