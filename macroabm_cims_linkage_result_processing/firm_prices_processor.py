"""Convert raw CIMS price CSVs into firm_prices.csv for FirmExogenousPriceSetter.

Usage (CLI)
-----------
    python -m macroabm_cims_linkage_result_processing.firm_prices_processor \\
        --fossil  raw_data/fuels_cims/fuels_CIMS.csv \\
        --elec    "raw_data/fuels_cims/CIMS_exogenous prices_01_BC.csv" \\
        --output  raw_data/cims_prices/firm_prices.csv

Output format
-------------
A CSV with years as the row index and ISIC Rev. 4 industry codes as
column headers.  The values are normalised price indices (1.0 at
initial_year, default 2014) so that FirmExogenousPriceSetter can apply
them directly without unit-conversion concerns.

    year,B05,C19,D
    2000,0.72,0.58,0.68
    2005,0.81,0.73,0.74
    2010,0.93,0.97,0.89
    2015,1.08,1.05,1.11
    ...

ISIC mapping (defaults)
-----------------------
B05  Mining and quarrying
    Fossil CSV fuels: Coal, Natural Gas
    Plus a hardcoded petroleum-crude series (historical/IEA forecast).
    Each fuel is normalised to its own 2014 value, then the group is
    averaged so that unit differences ($/GJ vs $/barrel) cancel.

C19  Coke and refined petroleum products
    Fossil CSV fuels: Coke, Diesel, Fuel Oil, Gasoline, Jet Fuel
    Same normalise-then-average approach.

D   Electricity, gas, steam and air conditioning supply
    Electricity CSV fuel: Electricity
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from macroabm_cims_linkage_result_processing.cims_raw_reader import read_cims_price_series

# ---------------------------------------------------------------------------
# Petroleum crude (B05c): not exported by CIMS.
# Source: historical EIA Brent spot prices + IEA net-zero pathway projection.
# Unit: $/barrel — unit cancels when normalised to initial_year.
# ---------------------------------------------------------------------------
_CRUDE_YEARS: list[int] = [
    2014, 2016, 2018, 2019, 2020, 2022, 2023, 2025, 2030, 2035, 2040, 2045, 2050,
]
_CRUDE_PRICES_USD_BBL: list[float] = [
    61.0, 53.0, 81.0, 72.0, 47.0, 101.0, 86.0, 80.0, 75.0, 75.0, 75.0, 75.0, 75.0,
]

# ---------------------------------------------------------------------------
# Default mapping: CIMS fossil-fuel name → ISIC code
# ---------------------------------------------------------------------------
DEFAULT_FOSSIL_FUEL_TO_ISIC: dict[str, str] = {
    "Coal":        "B05",
    "Natural Gas": "B05",
    "Coke":        "C19",
    "Diesel":      "C19",
    "Fuel Oil":    "C19",
    "Gasoline":    "C19",
    "Jet Fuel":    "C19",
}

# Electricity is read from a separate CSV.
# The CIMS file only provides an aggregate electricity price (no per-technology
# breakdown), so all five generation sub-sectors share the same trajectory —
# matching the original linkage-branch behaviour where D01b–D01e were set equal
# to D01a.
_ELEC_FUEL_NAME = "Electricity"
_ELEC_ISIC_SUBSECTORS: list[str] = ["D01a", "D01b", "D01c", "D01d", "D01e"]


def _normalise_series(series: pd.Series, initial_year: int) -> pd.Series:
    """Return series / series(initial_year), interpolating if needed."""
    years = series.index.astype(float).values
    prices = series.values.astype(float)
    fn = interp1d(years, prices, bounds_error=False, fill_value="extrapolate")
    base = float(fn(initial_year))
    if base == 0.0:
        raise ValueError(
            f"Price series '{series.name}' has a zero value at {initial_year}; "
            "cannot normalise."
        )
    return pd.Series(prices / base, index=series.index, name=series.name)


def build_firm_prices_csv(
    fossil_csv_path: Path | str,
    elec_csv_path: Path | str,
    initial_year: int = 2014,
    fossil_fuel_to_isic: Optional[dict[str, str]] = None,
    include_petroleum_crude: bool = True,
    output_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Build the firm_prices.csv DataFrame from CIMS raw CSVs.

    Each fuel series is normalised to 1.0 at *initial_year* before
    averaging within an ISIC group, so that unit differences across fuels
    ($/GJ vs $/barrel) do not distort the combined trajectory.

    Args:
        fossil_csv_path: Path to the fossil fuels CIMS CSV
            (``fuels_CIMS.csv``).
        elec_csv_path: Path to the electricity CIMS CSV
            (``CIMS_exogenous prices_01_BC.csv`` or similar).
        initial_year: Base year for price normalisation (default 2014).
        fossil_fuel_to_isic: Override the default CIMS fuel → ISIC mapping.
            Keys must be fuel names as they appear in the CSV Branch column
            (last dot-segment, e.g. ``"Natural Gas"``).
        include_petroleum_crude: Whether to include the hardcoded petroleum-
            crude series in the B05 average (default True).
        output_path: If given, write the result to this path as a CSV.

    Returns:
        DataFrame with integer years as index and ISIC codes as columns.
        All values are normalised (1.0 at *initial_year*).
    """
    if fossil_fuel_to_isic is None:
        fossil_fuel_to_isic = DEFAULT_FOSSIL_FUEL_TO_ISIC

    fossil_series = read_cims_price_series(fossil_csv_path)
    elec_series = read_cims_price_series(elec_csv_path)

    # Collect normalised series grouped by ISIC code.
    # groups[isic] = list of normalised pd.Series (all on same year grid)
    groups: dict[str, list[pd.Series]] = {}

    for fuel_name, isic_code in fossil_fuel_to_isic.items():
        if fuel_name not in fossil_series:
            continue
        norm = _normalise_series(fossil_series[fuel_name], initial_year)
        groups.setdefault(isic_code, []).append(norm)

    # Petroleum crude (B05c): use hardcoded series
    if include_petroleum_crude:
        crude = pd.Series(
            _CRUDE_PRICES_USD_BBL,
            index=_CRUDE_YEARS,
            name="Petroleum Crude",
        )
        norm_crude = _normalise_series(crude, initial_year)
        groups.setdefault("B05", []).append(norm_crude)

    # Electricity — replicate the single aggregate series across all sub-sectors.
    if _ELEC_FUEL_NAME in elec_series:
        norm_elec = _normalise_series(elec_series[_ELEC_FUEL_NAME], initial_year)
        for subsector in _ELEC_ISIC_SUBSECTORS:
            groups.setdefault(subsector, []).append(norm_elec)

    # Average within each ISIC group onto a common year grid.
    # The common grid is the union of all year indices present.
    all_years: set[int] = set()
    for series_list in groups.values():
        for s in series_list:
            all_years.update(s.index.tolist())
    years_sorted = sorted(all_years)

    data: dict[str, list[float]] = {}
    for isic_code, series_list in groups.items():
        col_values: list[float] = []
        for yr in years_sorted:
            # Interpolate each series at yr, then average.
            vals: list[float] = []
            for s in series_list:
                idx = s.index.astype(float).values
                fn = interp1d(idx, s.values.astype(float),
                              bounds_error=False, fill_value="extrapolate")
                vals.append(float(fn(yr)))
            col_values.append(float(np.mean(vals)))
        data[isic_code] = col_values

    df = pd.DataFrame(data, index=years_sorted)
    df.index.name = "year"
    df = df.sort_index()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path)

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Convert CIMS raw CSVs to firm_prices.csv for FirmExogenousPriceSetter."
    )
    parser.add_argument(
        "--fossil",
        required=True,
        help="Path to the fossil fuels CIMS CSV (fuels_CIMS.csv).",
    )
    parser.add_argument(
        "--elec",
        required=True,
        help='Path to the electricity CIMS CSV (e.g. "CIMS_exogenous prices_01_BC.csv").',
    )
    parser.add_argument(
        "--output",
        default="raw_data/cims_prices/firm_prices.csv",
        help="Output path for firm_prices.csv (default: raw_data/cims_prices/firm_prices.csv).",
    )
    parser.add_argument(
        "--initial-year",
        type=int,
        default=2014,
        help="Base year for normalisation (default: 2014).",
    )
    parser.add_argument(
        "--no-crude",
        action="store_true",
        help="Exclude the hardcoded petroleum-crude series from the B05 average.",
    )
    args = parser.parse_args()

    df = build_firm_prices_csv(
        fossil_csv_path=args.fossil,
        elec_csv_path=args.elec,
        initial_year=args.initial_year,
        include_petroleum_crude=not args.no_crude,
        output_path=args.output,
    )
    print(f"Written {args.output}  ({len(df)} rows × {len(df.columns)} industries)")
    print(df.round(4).to_string())


if __name__ == "__main__":
    _cli()
