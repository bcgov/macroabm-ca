"""Unified simulation runner for any country or subnational region.

Usage:
    uv run python run_simulation.py                    # France (default)
    uv run python run_simulation.py DEU                # Germany (EU - no proxy needed)
    uv run python run_simulation.py CAN                # Canada (auto-detects disaggregated if raw_data present)
    uv run python run_simulation.py CAN --aggregated   # Canada with 18 aggregated industries
    uv run python run_simulation.py CAN --proxy DEU    # Canada with Germany as proxy
    uv run python run_simulation.py USA --proxy FRA    # USA with France as proxy
    uv run python run_simulation.py GBR --proxy DEU    # UK with Germany as proxy
    uv run python run_simulation.py CAN --scale 5000   # Custom scale
    uv run python run_simulation.py CAN_BC             # British Columbia (auto progressive PIT)
    uv run python run_simulation.py CAN_ON --scale 2000

CAN_BC automatically activates BC's progressive personal income tax
schedule from ``spoof_data/freda/BC_PIT_2014.csv`` with compound
CPI inflation indexing.

If ``raw_data/icio/icio_can_2014_disagg.csv`` exists, Canada and its
regions use 46 disaggregated industries by default.  Otherwise they
fall back to 18 aggregated sectors.  Use --aggregated to force the
18-sector mode regardless.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

from macro_data import DataWrapper
from macro_data.configuration_utils import default_data_configuration
from macro_data.readers.taxation.personal_income_tax.pit_schedule import PITSchedule
from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.configurations.central_government_configuration import CentralGovernmentConfiguration
from macromodel.simulation import Simulation


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def main():
    total_start = time.perf_counter()
    
    parser = argparse.ArgumentParser(description="Run macroeconomic simulation for a country.")
    parser.add_argument("country", nargs="?", default="FRA", help="Country code (default: FRA)")
    parser.add_argument("--proxy", default=None, help="Proxy country for non-EU countries (default: FRA for non-EU)")
    parser.add_argument("--scale", type=int, default=10_000, help="Scale factor (default: 10000)")
    parser.add_argument("--t-max", type=int, default=20, help="Number of timesteps (default: 20)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--output", default="output", help="Output directory (default: output)")
    parser.add_argument(
        "--aggregated", action="store_true", default=False,
        help="Force aggregated industries (18 sectors) even when disaggregated data is available. "
             "Applies to Canada and its regions (CAN_BC, CAN_ON, etc.)."
    )
    args = parser.parse_args()

    country = args.country.upper()
    proxy = args.proxy.upper() if args.proxy else None

    # Detect region codes (e.g. "CAN_BC" → parent "CAN")
    is_region = "_" in country
    parent_country = country.split("_")[0] if is_region else country

    # Resolve paths
    repo_root = Path(__file__).resolve().parent.parent
    raw_data_dir = repo_root / "raw_data"
    sample_data_dir = repo_root / "tests/test_macro_data/unit/sample_raw_data"
    raw_data = raw_data_dir if raw_data_dir.exists() else sample_data_dir
    output_dir = repo_root / args.output
    output_dir.mkdir(exist_ok=True)

    # Determine if country needs proxy (non-EU countries)
    # EU28 countries as of 2014 (includes UK which was still EU member)
    eu_countries = {
        "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA",
        "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD",
        "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE", "GBR"
    }
    
    needs_proxy = parent_country not in eu_countries
    proxy_country = proxy or ("FRA" if needs_proxy else None)

    print(f"=== Running simulation for {country} ===")
    if is_region:
        print(f"  Region of: {parent_country}")
    print(f"  Scale: {args.scale:,}")
    print(f"  Timesteps: {args.t_max}")
    print(f"  Seed: {args.seed}")
    
    if needs_proxy:
        print(f"  Proxy country: {proxy_country} (required for non-EU)")
    else:
        print(f"  Proxy country: {proxy_country or 'None (EU country)'}")

    # Build proxy dict — keyed by parent country
    if proxy_country and needs_proxy:
        proxy_dict = {parent_country: proxy_country}
    else:
        proxy_dict = {}

    # Auto-detect whether disaggregated Canada ICIO data is available.
    # Falls back to 18 aggregated industries when the raw_data folder is
    # absent (e.g. CI / sample data).  --aggregated overrides to 18 sectors
    # even when the full data is present.
    disagg_cio_path = raw_data_dir / "icio" / "icio_can_2014_disagg.csv"
    disagg_available = disagg_cio_path.exists()

    is_canada = parent_country == "CAN"
    use_disagg_can = (
        is_canada
        and proxy_country in eu_countries
        and disagg_available
        and not args.aggregated
    )

    if is_canada:
        if args.aggregated:
            if disagg_available:
                warnings.warn(
                    "Disaggregated Canada ICIO data is available but --aggregated "
                    "forces 18 aggregated sectors instead of 46.",
                    UserWarning,
                )
            print("  Note: Forcing aggregated Canada configuration (--aggregated)")
        elif use_disagg_can:
            print("  Note: Using disaggregated Canada energy sector reader (auto-detected)")
        elif not disagg_available:
            warnings.warn(
                f"Disaggregated Canada ICIO data not found at {disagg_cio_path}. "
                f"Falling back to 18 aggregated sectors.",
                UserWarning,
            )
            print("  Note: Disaggregated data not found — using standard Canada configuration")
    elif args.aggregated:
        warnings.warn(
            "--aggregated flag has no effect for non-Canada countries. "
            f"'{country}' is not a Canada run.",
            UserWarning,
        )

    # 1. Data preprocessing
    print("\n[1/4] Preprocessing data...")
    step_start = time.perf_counter()
    
    if is_region:
        # ── Subnational region path ──
        # For a single region, run the parent country but label output
        # with the region name.  Multi-region provincial simulations
        # require the full aggregation_structure + provincial reader
        # pipeline (see conftest examples).
        print(f"  Running as parent country {parent_country} (region label: {country})")
        data_config = default_data_configuration(
            countries=[parent_country],
            proxy_country_dict=proxy_dict if proxy_dict else None,
            scale={parent_country: args.scale},
            seed=args.seed,
            use_disagg_can_2014_reader=use_disagg_can,
            aggregate_industries=not use_disagg_can,
        )
    else:
        # ── Country path (backward compatible) ──
        data_config = default_data_configuration(
            countries=[country],
            proxy_country_dict=proxy_dict if proxy_dict else None,
            scale={country: args.scale},
            seed=args.seed,
            use_disagg_can_2014_reader=use_disagg_can,
            aggregate_industries=not use_disagg_can,
        )

    datawrapper = DataWrapper.from_config(data_config, raw_data, single_hfcs_survey=True)
    step_elapsed = time.perf_counter() - step_start
    print(f"  Created {datawrapper.n_industries} industries in {format_duration(step_elapsed)}")

    # 2. Simulation configuration
    step_start = time.perf_counter()
    print("\n[2/4] Configuring simulation...")
    n_industries = datawrapper.n_industries

    if use_disagg_can:
        country_config = CountryConfiguration.n_industry_default(n_industries=n_industries)
    else:
        country_config = CountryConfiguration()

    # ── Progressive PIT (auto-activated for CAN_BC) ────────────
    if country == "CAN_BC":
        print(f"\n  Activating BC progressive PIT schedule ...")
        schedule = PITSchedule.from_name_with_cpi("BC_PIT_2014.csv")
        cpi_years = sorted(schedule.cpi_map)
        print(f"  CPI inflation data: {cpi_years[0]}–{cpi_years[-1]} ({len(cpi_years)} years)")
        thresholds, rates, _, _ = schedule.get_brackets(tax_year=schedule.base_year)

        brackets = [(float(thresholds[i]), float(rates[i])) for i in range(len(thresholds))]
        country_config.central_government = CentralGovernmentConfiguration(
            pit_brackets=brackets,
            pit_taxable_income_deductions=schedule.basic_deduction,
            functions=country_config.central_government.functions,
        )
        if schedule.basic_deduction is not None:
            print(f"  Basic personal amount (deduction from taxable income): ${schedule.basic_deduction:,.0f}")
        print(f"  Progressive PIT: {len(brackets)} brackets")
        for i, (thresh, rate) in enumerate(brackets):
            print(f"    Bracket {i+1}: up to {thresh:>12,.0f} @ {rate:.1%}")

    # For the region path, the datawrapper only has the parent country
    # but we label output files with the region name.
    sim_key = parent_country if is_region else country

    sim_config = SimulationConfiguration(
        country_configurations={sim_key: country_config},
        t_max=args.t_max,
        seed=args.seed,
    )
    step_elapsed = time.perf_counter() - step_start
    print(f"  Configuration complete in {format_duration(step_elapsed)}")

    # 3. Run simulation
    print("\n[3/4] Running simulation...")
    step_start = time.perf_counter()
    simulation = Simulation.from_datawrapper(
        datawrapper=datawrapper, 
        simulation_configuration=sim_config
    )

    # ── CPI-index PIT bracket thresholds every year ──────────
    # BC indexes its income-tax brackets to CPI every year.
    # We fire on month==1 (January) so the hook works correctly
    # regardless of the timestep increment (monthly / quarterly / annual).
    if country == "CAN_BC":
        _schedule = schedule          # capture for closure
        _sim_key = sim_key
        def _index_pit_brackets(_sim, t, year, month):
            if t > 0 and month == 1:
                cg = _sim.countries[_sim_key].central_government
                cg.step_pit_brackets(
                    tax_year=year,
                    cpi_map=_schedule.cpi_map,
                    base_year=_schedule.base_year,
                )
        simulation.posthooks.append(_index_pit_brackets)

    simulation.run()
    step_elapsed = time.perf_counter() - step_start
    print(f"  Simulation complete in {format_duration(step_elapsed)}")
    print(f"  Average per timestep: {step_elapsed / args.t_max:.2f}s")

    # 4. Save results
    print("\n[4/4] Saving results...")
    sim_file = output_dir / f"sim_{country.lower()}.h5"
    shallow_file = output_dir / f"sim_{country.lower()}_shallow.h5"
    
    simulation.save(save_dir=output_dir, file_name=sim_file.name)
    simulation.shallow_hdf_save(save_dir=output_dir, file_name=shallow_file.name)
    
    print(f"  Full output: {sim_file}")
    print(f"  Shallow output: {shallow_file}")

    # Print summary — use parent_country for region runs
    results = simulation.shallow_df_dict()
    summary_key = parent_country if is_region else country
    print(f"\n=== Summary for {country} ===")
    if summary_key in results:
        print(results[summary_key].head())
    else:
        print(f"  (No summary for '{summary_key}'; available: {sorted(results.keys())})")

    # Final timing
    total_elapsed = time.perf_counter() - total_start
    print(f"\n=== Timing Summary ===")
    print(f"  Total elapsed: {format_duration(total_elapsed)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
