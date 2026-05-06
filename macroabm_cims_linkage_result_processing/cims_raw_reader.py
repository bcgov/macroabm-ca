"""Reader for raw CIMS exogenous price CSV files.

CIMS CSV file layout
--------------------
Line 1  Navigation row (",--> Navigate by typing to search,...").
        pandas treats this as the column header row when header=0 (default).
Line 2  Actual column headers: Branch, Type, Region, ..., 2000, 2005, ..., 2050, Comments.
        This becomes df.iloc[0] after pd.read_csv.
Line 3+ Data rows (df.iloc[1], df.iloc[2], ...).

Column positions (0-indexed)
-----------------------------
0   Branch  (e.g. "CIMS.Generic Fuels.Natural Gas")
6   Parameter  (e.g. "LCC_financial")
11  Unit  (e.g. "$/GJ")
12-22  Price data for years 2000, 2005, 2010, 2015, 2020, 2025, 2030, 2035, 2040, 2045, 2050

Year values are read from df.iloc[0, 12:23] and cast to int.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_HEADER_ROW = 0        # df.iloc index of the actual-header / year row
_YEAR_COL_START = 12
_YEAR_COL_END = 23     # exclusive slice endpoint (cols 12-22 inclusive)
_BRANCH_COL = 0
_PARAM_COL = 6
_UNIT_COL = 11


def read_cims_price_series(csv_path: Path | str) -> dict[str, pd.Series]:
    """Return all LCC_financial $/GJ price series from a CIMS CSV.

    Args:
        csv_path: Path to the CIMS CSV file (fuels_CIMS.csv or
            CIMS_exogenous prices_*.csv).

    Returns:
        Dict mapping fuel name (last dot-separated segment of the Branch
        column, e.g. "Natural Gas", "Electricity") to a pd.Series whose
        index contains the integer years (2000, 2005, ..., 2050) and whose
        values are the LCC_financial prices in $/GJ.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, header=0, dtype=str)

    # Row 0 holds the actual column-header row; extract year integers from it.
    year_row = df.iloc[_HEADER_ROW, _YEAR_COL_START:_YEAR_COL_END]
    years = year_row.astype(float).astype(int).values  # [2000, 2005, ..., 2050]

    result: dict[str, pd.Series] = {}
    for i in range(1, len(df)):
        row = df.iloc[i]
        param = str(row.iloc[_PARAM_COL]).strip()
        unit = str(row.iloc[_UNIT_COL]).strip()
        if param != "LCC_financial" or unit != "$/GJ":
            continue
        branch = str(row.iloc[_BRANCH_COL]).strip()
        fuel_name = branch.split(".")[-1]   # last segment, e.g. "Natural Gas"
        prices = row.iloc[_YEAR_COL_START:_YEAR_COL_END].astype(float).values
        result[fuel_name] = pd.Series(prices, index=years, name=fuel_name)

    return result
