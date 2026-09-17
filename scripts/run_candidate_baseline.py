"""Minimal turnkey runner for the provisional candidate growth baseline.

Loads a provincial DataWrapper pickle, applies the candidate growth baseline (with the
bundled observed provincial labour paths), runs the simulation, and prints the headline
national result (double-deflated real GVA growth).

This is a thin, self-contained example: the switches come from
`macromodel.configurations.growth_baseline_preset`; nothing here imports the validation
workspace (`dev/`).

Usage:
    uv run python scripts/run_candidate_baseline.py [path/to/datawrapper.pkl] [--quarters 53] [--seed 0] [--legacy]

  --legacy : run shipped defaults instead of the candidate baseline (for A/B comparison).

DATA: the DataWrapper pickle is NOT bundled (it is large and rebuildable from raw_data via
the tracked macro_data pipeline). Default search:
    <repo>/dev/pkl_files/disagg_sectorprovs_2026_07_10_default.pkl
Pass a path if yours lives elsewhere.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from macro_data import DataWrapper
from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.configurations.growth_baseline_preset import apply_candidate_growth_baseline
from macromodel.simulation import Simulation

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PKL = REPO / "dev/pkl_files/disagg_sectorprovs_2026_07_10_default.pkl"
HH_COLS = [
    "Real Household Consumption (Value)",
    "Household Consumption (Value)",
    "Real Household Investment (Value)",
    "Household Investment (Value)",
]
HOUSEHOLD_DEMAND_GROWTH = 0.02  # common 2% exogenous household-demand overlay (candidate arm)


def _extend_exogenous_national_accounts(model, required_length: int) -> None:
    """Hold the last real quarter of the exogenous national-accounts series flat once the
    IMF/OECD data (~2023Q4) runs out, so the exogenous setters can be indexed to the full
    horizon. Self-contained (mirrors the dev build helper) so the runner needs no dev/ code.
    """
    for country in model.countries.values():
        frame = country.exogenous.national_accounts_during.copy()
        if len(frame) >= required_length:
            continue
        last_index = frame.index[-1]
        rows, index = [], []
        for step in range(required_length - len(frame)):
            rows.append(frame.iloc[-1].copy())
            index.append(last_index + pd.DateOffset(months=3 * (step + 1)))
        country.exogenous.national_accounts_during = pd.concat([frame, pd.DataFrame(rows, index=index)], axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl", nargs="?", default=str(DEFAULT_PKL))
    ap.add_argument("--quarters", type=int, default=53)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--legacy", action="store_true", help="run shipped defaults (no candidate overrides)")
    args = ap.parse_args()

    pkl = Path(args.pkl)
    if not pkl.exists():
        raise SystemExit(
            f"DataWrapper pickle not found: {pkl}\n"
            "Pass a path, or rebuild it from raw_data via the macro_data pipeline."
        )

    data = DataWrapper.init_from_pickle(pkl)
    # province keys are the CAN_* countries (ROW is the rest-of-world block, not a province)
    country_names = [c for c in data.all_country_names if c.startswith("CAN_")]
    if not country_names:
        raise SystemExit("Could not determine province keys from the DataWrapper.")

    cfg = SimulationConfiguration(
        seed=args.seed,
        country_configurations={
            c: CountryConfiguration.n_industry_default(n_industries=data.n_industries) for c in country_names
        },
        t_max=args.quarters,
    )
    if not args.legacy:
        for i, c in enumerate(country_names):
            apply_candidate_growth_baseline(
                cfg.country_configurations[c],
                use_observed_labour_path=True,
                province=c,
                n_quarters=args.quarters + 1,
                # per-province demography seed (matches the documented run convention)
                demography_seed=1000 + i + 100 * args.seed,
            )

    m = Simulation.from_datawrapper(datawrapper=data, simulation_configuration=cfg)

    if not args.legacy:
        # candidate baseline uses the exogenous government/household paths (held flat past
        # the data tail) and the common 2% household-demand overlay.
        _extend_exogenous_national_accounts(m, required_length=args.quarters + 1)
        for country in m.countries.values():
            fr = country.exogenous.national_accounts_during
            fac = (1.0 + HOUSEHOLD_DEMAND_GROWTH) ** (np.arange(len(fr)) / 4.0)
            for col in HH_COLS:
                if col in fr.columns:
                    fr[col] = fr[col].values * fac

    p0 = {c: np.array(m.countries[c].economy.ts.current("good_prices"), float).reshape(-1) for c in country_names}
    ro = np.zeros(args.quarters)
    ri = np.zeros(args.quarters)
    u = np.zeros(args.quarters)
    for t in range(args.quarters):
        m.iterate(t)
        for c in country_names:
            f = m.countries[c].firms
            base = p0[c]
            qr = np.array(f.ts.current("production"), float)
            ui = np.array(f.ts.current("used_intermediate_inputs"), float)
            ro[t] += float((qr * base).sum())
            ri[t] += float((ui * base[None, :]).sum()) if ui.ndim == 2 else 0.0
        u[t] = float(
            np.mean(
                [
                    np.array(m.countries[c].economy.ts.current("unemployment_rate"), float).reshape(-1)[0]
                    for c in country_names
                ]
            )
        )
    rva = ro - ri
    yrs = (args.quarters - 1) / 4.0
    ann = ((rva[-1] / rva[0]) ** (1 / yrs) - 1) * 100

    mode = "LEGACY (shipped defaults)" if args.legacy else "CANDIDATE growth baseline"
    print(f"\n=== {mode} | {len(country_names)} provinces | seed {args.seed} | {args.quarters}q ===")
    print(
        f"  double-deflated real GVA: {rva[0] / 1e9:.1f}B -> {rva[-1] / 1e9:.1f}B  "
        f"({(rva[-1] / rva[0] - 1) * 100:+.1f}% cumulative, {ann:+.2f}%/yr)"
    )
    print(f"  unemployment: {u[0] * 100:.1f}% -> {u[-1] * 100:.1f}%")
    print("  (provisional baseline; national aggregate only; not validated for quantitative inference)")


if __name__ == "__main__":
    main()
