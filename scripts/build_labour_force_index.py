"""Build the bundled provincial labour-force index from StatCan LFS 14-10-0327.

Reads the raw LFS CSV and writes a small annual index (base 1.0 at 2014) per province to
`scripts/data/labour_force_index_2014_2024.json`. That JSON is bundled on the branch so
the candidate baseline runs turnkey; this script only needs to be re-run if the LFS data
is updated.

The runtime interpolation (annual -> quarterly, held flat after the last year) lives in
`macromodel.configurations.growth_baseline_preset.observed_labour_force_index`, so the
model side needs only the small JSON, not this script or the ~72 MB raw CSV.

Usage:
    uv run python scripts/build_labour_force_index.py [path/to/14100327.csv]

CSV search order if no path is given:
    <repo>/../raw_data/14100327.csv   (shared SESIT raw_data)
    <repo>/dev/statcan/14100327.csv   (validation workspace)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "scripts/data/labour_force_index_2014_2024.json"
Y0, Y1 = 2014, 2024

PROVINCE_NAMES = {
    "CAN_AB": "Alberta",
    "CAN_BC": "British Columbia",
    "CAN_MB": "Manitoba",
    "CAN_NB": "New Brunswick",
    "CAN_NL": "Newfoundland and Labrador",
    "CAN_NS": "Nova Scotia",
    "CAN_ON": "Ontario",
    "CAN_PE": "Prince Edward Island",
    "CAN_QC": "Quebec",
    "CAN_SK": "Saskatchewan",
}


def _find_csv(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    for c in (REPO.parent / "raw_data" / "14100327.csv", REPO / "dev/statcan/14100327.csv"):
        if c.exists():
            return c
    raise FileNotFoundError("LFS 14-10-0327 CSV not found. Pass its path, or place 14100327.csv in ../raw_data/.")


def build(csv_path: Path) -> dict[str, dict[str, float]]:
    df = pd.read_csv(
        csv_path,
        low_memory=False,
        usecols=["REF_DATE", "GEO", "Labour force characteristics", "Gender", "Age group", "VALUE"],
    )
    gender = df["Gender"].str.strip().str.lower()
    df = df[
        (df["Labour force characteristics"] == "Labour force")
        & (df["Age group"] == "15 years and over")
        & (gender.isin(["both sexes", "total - gender", "total - genders"]))
        & (df["GEO"].isin(PROVINCE_NAMES.values()))
    ]
    annual = df.pivot_table(index="REF_DATE", columns="GEO", values="VALUE").loc[Y0:Y1]
    out: dict[str, dict[str, float]] = {}
    for prov, geo in PROVINCE_NAMES.items():
        s = annual[geo]
        out[prov] = {str(int(y)): float(s.loc[y] / s.loc[Y0]) for y in s.index}
    return out


if __name__ == "__main__":
    csv = _find_csv(sys.argv[1] if len(sys.argv) > 1 else None)
    idx = build(csv)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(idx, indent=2, sort_keys=True))
    print(f"read {csv}")
    print(f"wrote {OUT}")
    for p, d in idx.items():
        print(f"  {p}: {Y0}=1.000 -> {Y1}={d[str(Y1)]:.4f} ({(d[str(Y1)] - 1) * 100:+.1f}%)")
