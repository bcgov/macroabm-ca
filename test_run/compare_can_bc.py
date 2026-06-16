"""Compare CAN (flat tax) vs CAN_BC (progressive PIT) simulation results.

Runs both simulations locally and compares:
1. Effective income tax rate at each timestep (captured via posthook)
2. Income tax revenue over time
3. Key macro indicators (GDP, wages, unemployment, CPI)
4. Employee income distribution (Gini, percentiles)

If ``raw_data/icio/icio_can_2014_disagg.csv`` exists, both simulations use
46 disaggregated industries by default.  Use --aggregated to force 18 sectors.

Usage:
    uv run python compare_can_bc.py                     # run both, compare in-process
    uv run python compare_can_bc.py --no-run            # use existing HDF5 files
    uv run python compare_can_bc.py --t-max 10 --scale 5000  # faster test
    uv run python compare_can_bc.py --aggregated        # use 18 aggregated sectors
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from macro_data import DataWrapper
from macro_data.configuration_utils import default_data_configuration
from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.configurations.central_government_configuration import CentralGovernmentConfiguration
from macromodel.simulation import Simulation
from macro_data.readers.taxation.personal_income_tax.pit_schedule import PITSchedule


# ═════════════════════════════════════════════════════════════════════
# helpers
# ═════════════════════════════════════════════════════════════════════

def _gini(x: np.ndarray) -> float:
    x = np.sort(x[x > 0])
    n = len(x)
    if n == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * x) - (n + 1) * np.sum(x)) / (n * np.sum(x)))


def _tax_rate_collector(record: list[float], country_key: str):
    def hook(sim, t, year, month):
        record.append(float(sim.countries[country_key].central_government.states["Income Tax"]))
    return hook


# ═════════════════════════════════════════════════════════════════════
# run mode
# ═════════════════════════════════════════════════════════════════════

def run_simulation(country: str, args) -> tuple[Simulation, list[float]]:
    """Run one simulation and return (sim, effective_tax_rate_history)."""
    t0 = time.perf_counter()
    upper = country.upper()
    is_region = "_" in upper
    parent = upper.split("_")[0] if is_region else upper

    eu28 = {"AUT","BEL","BGR","HRV","CYP","CZE","DNK","EST","FIN","FRA",
            "DEU","GRC","HUN","IRL","ITA","LVA","LTU","LUX","MLT","NLD",
            "POL","PRT","ROU","SVK","SVN","ESP","SWE","GBR"}
    needs_proxy = parent not in eu28
    proxy = (args.proxy or "FRA").upper() if needs_proxy else None
    pdict = {parent: proxy} if proxy else {}

    # Auto-detect disaggregated Canada ICIO data (same logic as run_simulation.py).
    raw_data_dir = REPO_ROOT / "raw_data"
    disagg_cio_path = raw_data_dir / "icio" / "icio_can_2014_disagg.csv"
    disagg_available = disagg_cio_path.exists()

    is_canada = parent == "CAN"
    use_disagg = is_canada and disagg_available and not args.aggregated

    # Warn on user-visible scenarios (same pattern as run_simulation.py).
    if is_canada:
        if args.aggregated:
            if disagg_available:
                warnings.warn(
                    "Disaggregated Canada ICIO data is available but --aggregated "
                    "forces 18 aggregated sectors instead of 46.",
                    UserWarning,
                )
            print(f"  [{upper}] Forcing aggregated (18 sectors, --aggregated)")
        elif use_disagg:
            print(f"  [{upper}] Disaggregated (46 sectors, auto-detected)")
        elif not disagg_available:
            warnings.warn(
                f"Disaggregated Canada ICIO data not found at {disagg_cio_path}. "
                f"Falling back to 18 aggregated sectors.",
                UserWarning,
            )
            print(f"  [{upper}] Aggregated (18 sectors, data not found)")
    elif args.aggregated:
        warnings.warn(
            f"--aggregated flag has no effect for non-Canada countries. "
            f"'{upper}' is not a Canada run.",
            UserWarning,
        )

    msg = "PIT ON" if upper == "CAN_BC" else "flat"
    print(f"  [{upper}] scale={args.scale:,}  t_max={args.t_max}  tax={msg}")

    raw_dir = REPO_ROOT / "raw_data"
    sample_dir = REPO_ROOT / "tests/test_macro_data/unit/sample_raw_data"
    raw = raw_dir if raw_dir.exists() else sample_dir
    dc = default_data_configuration(
        countries=[parent], proxy_country_dict=pdict or None,
        scale={parent: args.scale}, seed=args.seed,
        use_disagg_can_2014_reader=use_disagg, aggregate_industries=not use_disagg,
    )
    dw = DataWrapper.from_config(dc, raw, single_hfcs_survey=True)

    cc = CountryConfiguration.n_industry_default(dw.n_industries) if use_disagg else CountryConfiguration()

    sch = None  # PITSchedule for CAN_BC (captured by CPI-indexing posthook)
    if upper == "CAN_BC":
        sch = PITSchedule.from_name_with_cpi("BC_PIT_2014.csv")
        thr, r, _, _ = sch.get_brackets(tax_year=sch.base_year)
        cc.central_government = CentralGovernmentConfiguration(
            pit_brackets=[(float(thr[i]), float(r[i])) for i in range(len(thr))],
            pit_credit_base=sch.basic_deduction,
            functions=cc.central_government.functions,
        )
        if sch.basic_deduction is not None:
            print(f"  [{upper}] basic_deduction=${sch.basic_deduction:,.0f} "
                  f"(credit @ {r[0]:.1%} = ${sch.basic_deduction * r[0]:,.0f})")
        print(f"  [{upper}] {len(thr)} PIT brackets loaded")

    sim_key = parent
    scfg = SimulationConfiguration(
        country_configurations={sim_key: cc}, t_max=args.t_max, seed=args.seed,
    )
    sim = Simulation.from_datawrapper(datawrapper=dw, simulation_configuration=scfg)

    rates: list[float] = []
    sim.posthooks.append(_tax_rate_collector(rates, sim_key))

    # ── CPI-index PIT bracket thresholds & basic_deduction every year ──
    if sch is not None:
        _schedule = sch
        _sim_key = sim_key
        def _index_pit_brackets(_sim, t, year, month):
            if t > 0 and month == 1:
                cg = _sim.countries[_sim_key].central_government
                cg.step_pit_brackets(
                    tax_year=year,
                    cpi_map=_schedule.cpi_map,
                    base_year=_schedule.base_year,
                )
        sim.posthooks.append(_index_pit_brackets)

    sim.run()
    print(f"  [{upper}] done in {time.perf_counter() - t0:.1f}s")
    return sim, rates


# ═════════════════════════════════════════════════════════════════════
# --no-run: load from HDF5
# ═════════════════════════════════════════════════════════════════════

def _read_ts(h5f, country: str, path: str) -> np.ndarray:
    """Read a timeseries dataset from HDF5, shape (t, ...) -> (t,)."""
    import h5py as _h5py
    ds = h5f[f"{country}/central_government/{path}"][:]
    if ds.ndim == 2:
        return ds.sum(axis=1)
    return ds.squeeze()


def compare_from_hdf5(args, out_dir: Path):
    """Compare CAN vs CAN_BC using existing HDF5 files."""
    import h5py as _h5py

    h5_can = REPO_ROOT / "output/sim_can.h5"
    h5_bc = REPO_ROOT / "output/sim_can_bc.h5"

    for p, label in [(h5_can, "CAN"), (h5_bc, "CAN_BC")]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run with: uv run python run_simulation.py {label}")
            return 1

    t_max = args.t_max

    with _h5py.File(h5_can, "r") as fc, _h5py.File(h5_bc, "r") as fb:
        inc_can = _read_ts(fc, "CAN", "taxes_income")[:t_max]
        inc_bc = _read_ts(fb, "CAN", "taxes_income")[:t_max]
        emp_can = fc["CAN/individuals/employee_income"][-1]
        emp_bc = fb["CAN/individuals/employee_income"][-1]

    steps = np.arange(min(len(inc_can), len(inc_bc)))

    # ── Tax revenue plot ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(steps, inc_can / 1e9, "o-", label="CAN (flat)")
    ax.plot(steps, inc_bc / 1e9, "s-", label="CAN_BC (progressive)")
    ax.set_xlabel("Timestep"); ax.set_ylabel("Revenue (B $)")
    ax.set_title("Income Tax Revenue"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    diff = (inc_bc - inc_can) / 1e9
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in diff]
    ax.bar(steps, diff, color=colors, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Timestep"); ax.set_ylabel("Δ Revenue (B $)")
    ax.set_title("CAN_BC − CAN Income Tax Revenue"); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "compare_tax_revenue.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: compare_tax_revenue.png")

    # ── Macro comparison (from shallow HDF5 if available) ───────────
    sh_path_can = REPO_ROOT / "output/sim_can_shallow.h5"
    sh_path_bc = REPO_ROOT / "output/sim_can_bc_shallow.h5"
    if sh_path_can.exists() and sh_path_bc.exists():
        import pandas as pd
        sh_can = pd.read_hdf(sh_path_can, key="CAN")
        sh_bc = pd.read_hdf(sh_path_bc, key="CAN")

        fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
        for (col, label), ax in zip([
            ("Production","Production"), ("Wages","Wages"),
            ("Household Consumption","Consumption"),
            ("Unemployment Rate","Unemployment Rate"),
            ("CPI","CPI Inflation"), ("Gross Output","Gross Output"),
        ], axes2.flat):
            if col not in sh_can.columns or col not in sh_bc.columns:
                ax.set_title(f"{label} (N/A)"); continue
            cv = sh_can[col].values[:t_max]; bv = sh_bc[col].values[:t_max]
            n = min(len(cv), len(bv))
            ax.plot(range(n), cv[:n], "o-", label="CAN flat", linewidth=2)
            ax.plot(range(n), bv[:n], "s-", label="CAN_BC prog", linewidth=2)
            ax.set_title(label); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(out_dir / "compare_macro.png", dpi=150)
        plt.close(fig2)
        print(f"  Saved: compare_macro.png")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CAN vs CAN_BC — Comparison (from HDF5)")
    print(f"  NOTE: effective tax rate not tracked in HDF5 — run without --no-run")
    print(f"{'='*60}")
    N = min(len(inc_can), len(inc_bc))
    print(f"\n  Income Tax Revenue t={N-1}:")
    print(f"    CAN flat:     {inc_can[N-1]/1e9:>12.2f} B$")
    print(f"    CAN_BC prog:  {inc_bc[N-1]/1e9:>12.2f} B$")
    print(f"    Delta:        {inc_bc[N-1]/1e9 - inc_can[N-1]/1e9:>+12.2f} B$")
    print(f"\n  Income Distribution (final timestep):")
    print(f"    Gini:  CAN={_gini(emp_can):.4f}  CAN_BC={_gini(emp_bc):.4f}")
    print(f"    P50:   CAN={np.percentile(emp_can,50):,.0f}  CAN_BC={np.percentile(emp_bc,50):,.0f}")
    print(f"\n  Saved to: {out_dir}")
    return 0


# ═════════════════════════════════════════════════════════════════════
# in-process comparison
# ═════════════════════════════════════════════════════════════════════

def compare_in_process(sim_can, sim_bc, tax_can, tax_bc, args, out_dir):
    t_max = args.t_max
    can_key = "CAN"
    bc_key = "CAN" if "CAN_BC" not in sim_bc.countries else "CAN_BC"

    cg_can = sim_can.countries[can_key].central_government
    cg_bc = sim_bc.countries[bc_key].central_government

    can_sh = sim_can.countries[can_key].shallow_output()
    bc_sh = sim_bc.countries[bc_key].shallow_output()

    # ── Effective tax rate + revenue ────────────────────────────────
    steps = np.arange(len(tax_can))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(steps, np.array(tax_can)*100, "o-", label="CAN flat", linewidth=2)
    ax.plot(steps, np.array(tax_bc)*100, "s-", label="CAN_BC progressive", linewidth=2)
    ax.set_xlabel("Timestep"); ax.set_ylabel("Effective Rate (%)")
    ax.set_title("Effective Income Tax Rate"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    inc_c = cg_can.ts.get_aggregate("taxes_income")[:t_max]
    inc_b = cg_bc.ts.get_aggregate("taxes_income")[:t_max]
    ax.plot(range(len(inc_c)), inc_c/1e9, "o-", label="CAN")
    ax.plot(range(len(inc_b)), inc_b/1e9, "s-", label="CAN_BC")
    ax.set_xlabel("Timestep"); ax.set_ylabel("Revenue (B $)")
    ax.set_title("Income Tax Revenue"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    diff_rate = (np.array(tax_bc) - np.array(tax_can))*100
    ax.bar(steps, diff_rate, color=["#2ca02c" if d>=0 else "#d62728" for d in diff_rate], alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Timestep"); ax.set_ylabel("Δ (pp)")
    ax.set_title("CAN_BC − CAN Rate Diff"); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "compare_tax_rates.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: compare_tax_rates.png")

    # ── Themed macro timeseries (4 grids) ──────────────────────────
    # All columns from shallow_output() that exist in both DataFrames.
    _all_cols = [c for c in can_sh.columns if c in bc_sh.columns]

    # Helper: draw one themed grid, return list of (col, label) actually plotted.
    def _plot_themed_grid(metrics_map: list[tuple[str, str]],
                           filename: str, ncols: int = 3, figsize_scale: float = 1.0):
        found_m = [(col, label) for col, label in metrics_map if col in _all_cols]
        if not found_m:
            print(f"  (no matching metrics for {filename})")
            return []
        nr = (len(found_m) + ncols - 1) // ncols
        fig, axes = plt.subplots(nr, ncols, figsize=(6 * ncols * figsize_scale,
                                                       3.5 * nr * figsize_scale))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for j, (col, label) in enumerate(found_m):
            ax = axes[j]
            cv = can_sh[col].values[:t_max]
            bv = bc_sh[col].values[:t_max]
            n = min(len(cv), len(bv))
            ax.plot(range(n), cv[:n], "o-", label="CAN flat", linewidth=2, markersize=4)
            ax.plot(range(n), bv[:n], "s-", label="CAN_BC prog", linewidth=2, markersize=4)
            ax.set_title(label); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        for j in range(len(found_m), len(axes)):
            axes[j].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=150)
        plt.close(fig)
        print(f"  Saved: {filename} ({len(found_m)} metrics)")
        return found_m

    # Grid 1 — Consumption & Income
    consumption_metrics = [
        ("Household Consumption",   "Household Consumption"),
        ("Government Consumption",  "Government Consumption"),
        ("Wages",                   "Wages"),
        ("Profits",                 "Profits"),
        ("Sales",                   "Sales"),
        ("Operating Surplus",       "Operating Surplus"),
    ]
    cons_found = _plot_themed_grid(consumption_metrics,
                                   "compare_consumption_income.png")

    # Grid 2 — Investment & Credit
    # Omitted: Consumption Loan Debt, Mortgage Debt, Central Bank Policy Rate —
    # these are identical between CAN and CAN_BC because the PIT rate does not
    # influence credit-market clearing or monetary-policy decisions.
    investment_metrics = [
        ("Capital Bought",     "Capital Bought"),
        ("Inventory Changes",  "Inventory Changes"),
        ("Bought Input Costs", "Bought Input Costs"),
    ]
    inv_found = _plot_themed_grid(investment_metrics,
                                  "compare_investment_credit.png")

    # Grid 3 — Government & Prices
    govt_metrics = [
        ("Taxes Paid on Production","Taxes on Production"),
        ("Taxes on Products",       "Taxes on Products"),
        ("Unemployment Rate",       "Unemployment Rate"),
        ("CPI",                     "CPI Inflation"),
        ("PPI",                     "PPI Inflation"),
        ("CFPI",                    "CFPI Inflation"),
    ]
    govt_found = _plot_themed_grid(govt_metrics,
                                   "compare_government_prices.png")

    # Grid 4 — Trade & Output
    trade_metrics = [
        ("Imports",       "Imports"),
        ("Exports",       "Exports"),
        ("Production",    "Production"),
        ("Gross Output",  "Gross Output"),
        ("Used Input Costs","Used Input Costs"),
    ]
    trade_found = _plot_themed_grid(trade_metrics,
                                    "compare_trade_output.png")

    # ── Omitted (PIT-independent) metrics ───────────────────────────
    _pit_independent = [
        "Consumption Expansion Loan Debt",
        "Mortgage Debt",
        "Central Bank Policy Rate",
    ]
    _omitted = [c for c in _pit_independent if c in _all_cols]
    if _omitted:
        print(f"  Note: {', '.join(_omitted)} omitted from plots — identical between "
              f"CAN and CAN_BC because the PIT rate does not affect credit-market "
              f"clearing or monetary-policy decisions.")

    # ── CSV export (all shallow_output columns + tax extras) ────────
    import pandas as pd
    csv_rows = []
    for col in _all_cols:
        cv = can_sh[col].values[:t_max]
        bv = bc_sh[col].values[:t_max]
        n = min(len(cv), len(bv))
        for t in range(n):
            csv_rows.append({
                "metric": col, "timestep": t,
                "CAN_flat": cv[t], "CAN_BC_prog": bv[t],
                "delta": bv[t] - cv[t],
            })
    for t in range(min(len(tax_can), len(tax_bc))):
        csv_rows.append({
            "metric": "effective_income_tax_rate",
            "timestep": t,
            "CAN_flat": tax_can[t], "CAN_BC_prog": tax_bc[t],
            "delta": tax_bc[t] - tax_can[t],
        })
    for t in range(min(len(inc_c), len(inc_b))):
        csv_rows.append({
            "metric": "income_tax_revenue",
            "timestep": t,
            "CAN_flat": inc_c[t], "CAN_BC_prog": inc_b[t],
            "delta": inc_b[t] - inc_c[t],
        })
    csv_df = pd.DataFrame(csv_rows)
    csv_path = out_dir / "compare_can_bc_timeseries.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path.name} ({len(csv_df)} rows)")

    # ── Summary table ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CAN vs CAN_BC — Progressive PIT Comparison Summary")
    print(f"{'='*60}")
    print(f"\n  {'Metric':<30} {'CAN flat':>14} {'CAN_BC prog':>14} {'Δ':>12}")
    print(f"  {'─'*30} {'─'*14} {'─'*14} {'─'*12}")

    fc, fb = tax_can[-1]*100, tax_bc[-1]*100
    print(f"  {'Eff. Tax Rate (%)':<30} {fc:>14.2f} {fb:>14.2f} {fb-fc:>+11.2f}pp")

    N = min(len(inc_c), len(inc_b))
    print(f"  {'Tax Revenue (B$) t='+str(N-1):<30} {inc_c[N-1]/1e9:>14.2f} {inc_b[N-1]/1e9:>14.2f} {inc_b[N-1]/1e9-inc_c[N-1]/1e9:>+11.2f}")

    emp_c = sim_can.countries[can_key].individuals.states["Employee Income"]
    emp_b = sim_bc.countries[bc_key].individuals.states["Employee Income"]
    print(f"\n  Income Distribution (final timestep):")
    print(f"  {'Gini':<30} {_gini(emp_c):>14.4f} {_gini(emp_b):>14.4f}")
    for pct in [10, 50, 90]:
        print(f"  {f'P{pct}':<30} {np.percentile(emp_c,pct):>14,.0f} {np.percentile(emp_b,pct):>14,.0f}")

    print(f"\n  Effective Income Tax Rate by Timestep:")
    print(f"  {'Step':<6} {'CAN flat':>10} {'CAN_BC prog':>14} {'Δ (pp)':>10}")
    print(f"  {'─'*6} {'─'*10} {'─'*14} {'─'*10}")
    for i in range(min(len(tax_can), len(tax_bc))):
        d = (tax_bc[i] - tax_can[i]) * 100
        print(f"  {i:<6} {tax_can[i]*100:>10.2f}% {tax_bc[i]*100:>14.2f}% {d:>+9.2f}pp")

    print(f"\n  Plots + CSV saved to: {out_dir}")


# ═════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Compare CAN vs CAN_BC simulations")
    parser.add_argument("--scale", type=int, default=10_000)
    parser.add_argument("--t-max", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--out", default="output")
    parser.add_argument("--no-run", action="store_true",
                        help="Compare from existing HDF5 files in output/")
    parser.add_argument(
        "--aggregated", action="store_true", default=False,
        help="Force aggregated industries (18 sectors) even when disaggregated data is available.",
    )
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.no_run:
        print("Comparing from existing HDF5 files...")
        return compare_from_hdf5(args, out_dir)

    print("=" * 60)
    print("  CAN vs CAN_BC — Progressive PIT Comparison")
    print("=" * 60)

    sim_can, tax_can = run_simulation("CAN", args)
    sim_bc, tax_bc = run_simulation("CAN_BC", args)

    compare_in_process(sim_can, sim_bc, tax_can, tax_bc, args, out_dir)

    sim_can.save(save_dir=out_dir, file_name="sim_can.h5")
    sim_can.shallow_hdf_save(save_dir=out_dir, file_name="sim_can_shallow.h5")
    sim_bc.save(save_dir=out_dir, file_name="sim_can_bc.h5")
    sim_bc.shallow_hdf_save(save_dir=out_dir, file_name="sim_can_bc_shallow.h5")
    print(f"\n  Saved: sim_can.h5, sim_can_bc.h5 (+ shallow)")
    print("  Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
