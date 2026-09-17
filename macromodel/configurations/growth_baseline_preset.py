"""Candidate (provisional) real-growth baseline preset.

This module carries the machine-readable parameter set for the *provisional*
internally-transmitted real-growth configuration, and a helper that applies it to a
`CountryConfiguration`. Every field is an opt-in override; the shipped model defaults
are unchanged, so a configuration that does not call `apply_candidate_growth_baseline`
reproduces legacy behaviour exactly.

STATUS: provisional. This baseline is robust across five seeds and suitable for model
development and controlled comparative scenario testing, but is NOT yet fully validated
for quantitative inference. No decision has been made about merging to MacroABM-CA
`main` or upstreaming general changes to INET.

DATA DEPENDENCY (not committed here): the baseline additionally requires
  (1) a provincial DataWrapper pickle (e.g. dev/pkl_files/disagg_sectorprovs_*.pkl), and
  (2) an observed provincial labour-force index path (derived from StatCan LFS 14-10-0327).
Neither is bundled in this branch. `CANDIDATE_GROWTH_BASELINE["labour_force_index"]` is
left None; a runner must supply a per-province quarterly index (base 1.0 at t0). Absent a
path, demography falls back to the shipped `NoAging` default (fixed labour force).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Bundled observed provincial labour-force index (annual, base 1.0 at 2014), built from
# StatCan LFS 14-10-0327 by scripts/build_labour_force_index.py. Small (~4 KB) so the
# candidate baseline runs turnkey without the ~72 MB raw CSV.
_LABOUR_INDEX_JSON = Path(__file__).resolve().parents[2] / "scripts/data/labour_force_index_2014_2024.json"
_LABOUR_BASE_YEAR = 2014


def observed_labour_force_index(
    n_quarters: int, province: Optional[str] = None, uniform: bool = False
) -> dict[str, list[float]] | list[float]:
    """Quarterly observed labour-force index (base 1.0 at t0), interpolated to n_quarters.

    Reads the bundled annual index and linearly interpolates annual -> quarterly, holding
    the last observed year flat thereafter (mirrors the exogenous national-accounts tail).

    Args:
        n_quarters: number of quarters required (e.g. t_max + 1).
        province: a country key (e.g. "CAN_ON"); if given, returns that province's list,
            else a dict of all provinces.
        uniform: if True, every province gets the national (sum-of-provinces) index -- the
            control that removes province-specific labour variation.

    Raises:
        FileNotFoundError: if the bundled JSON is missing (rebuild with
            scripts/build_labour_force_index.py).
    """
    if not _LABOUR_INDEX_JSON.exists():
        raise FileNotFoundError(f"{_LABOUR_INDEX_JSON} not found; rebuild with scripts/build_labour_force_index.py")
    raw = json.loads(_LABOUR_INDEX_JSON.read_text())
    years = sorted(int(y) for y in next(iter(raw.values())))
    year_q = np.array([(y - _LABOUR_BASE_YEAR) * 4 + 1.5 for y in years])  # annual obs at mid-year
    grid = np.arange(n_quarters, dtype=float)

    def _interp(annual_by_year: dict[str, float]) -> list[float]:
        vals = np.array([annual_by_year[str(y)] for y in years], dtype=float)
        idx = np.interp(grid, year_q, vals)  # np.interp holds the endpoints flat outside the range
        return (idx / idx[0]).tolist()

    if uniform:
        # national index = provinces weighted equally in index space is not meaningful;
        # reconstruct the national sum requires levels, which the index drops. The uniform
        # control instead applies the cross-province mean index to every province.
        mean_by_year = {str(y): float(np.mean([raw[p][str(y)] for p in raw])) for y in years}
        shared = _interp(mean_by_year)
        result = {p: list(shared) for p in raw}
    else:
        result = {p: _interp(raw[p]) for p in raw}

    if province is not None:
        return result[province]
    return result


# Machine-readable parameter set (opt-in overrides on n_industry_default).

# Machine-readable parameter set (opt-in overrides on n_industry_default).
CANDIDATE_GROWTH_BASELINE: dict[str, Any] = {
    "demand_estimator": {
        # reconnect the endogenous growth forecast (shipped default is 0.0 = discarded)
        "sectoral_growth_adjustment_speed": 1.0,
        "firm_growth_adjustment_speed": 1.0,
        # adaptive-expectations smoothing alpha (shipped default 1.0 = no smoothing)
        "demand_smoothing": 0.3,
    },
    "demand_for_goods": {
        # weight rho on recorded unmet demand (shipped default 1.0)
        "unmet_demand_weight": 0.25,
    },
    "excess_demand": {
        # record demand-side unmet orders rather than seller spare capacity
        # (shipped default 1.0 = capacity-capped)
        "consider_capital_inputs": 0.0,
    },
    "target_production": {
        "capital_inputs_target_considers_capital_inputs": 1.0,  # w (shipped default)
    },
    "target_capital_inputs": {
        "target_capital_inputs_fraction": 0.1,  # lambda (shipped default 0.0)
        "rolling_reference": True,  # shipped default False
    },
    "government_consumption_setter": "ExogenousGovernmentConsumptionSetter",
    "household_consumption_setter": "ExogenousHouseholdConsumption",
    "household_investment_setter": "ExogenousHouseholdInvestment",
    "demography": "ExogenousLabourForcePath",  # shipped default "NoAging"
    # supplied per-province by the runner; None -> falls back to NoAging (see module docstring)
    "labour_force_index": None,
}


def apply_candidate_growth_baseline(
    country_config,
    labour_force_index: Optional[list[float]] = None,
    demography_seed: int = 0,
    use_observed_labour_path: bool = False,
    province: Optional[str] = None,
    n_quarters: Optional[int] = None,
    uniform_labour_path: bool = False,
):
    """Apply the candidate growth baseline (opt-in overrides) to one CountryConfiguration.

    The labour-supply path is an on/off switch like the other mechanisms, with three ways
    to set it (in priority order):

    1. `labour_force_index=[...]`  -- pass an explicit per-province quarterly index.
    2. `use_observed_labour_path=True` (+ `province`, `n_quarters`) -- load the bundled
       observed provincial index (StatCan LFS 14-10-0327) for that province and horizon.
    3. neither -- demography stays at the shipped `NoAging` default (fixed labour force).

    Args:
        country_config: a CountryConfiguration (e.g. CountryConfiguration.n_industry_default(...)).
        labour_force_index: explicit quarterly index (base 1.0 at t0); overrides the flag.
        demography_seed: reproducible-selection seed for the labour-force reclassification.
        use_observed_labour_path: load the bundled observed index for `province`.
        province: country key (e.g. "CAN_ON"), required with use_observed_labour_path.
        n_quarters: horizon (t_max + 1), required with use_observed_labour_path.
        uniform_labour_path: with the flag, use the cross-province mean index (control arm).

    Returns:
        The mutated country_config (also mutated in place).
    """
    p = CANDIDATE_GROWTH_BASELINE
    fc = country_config.firms.functions
    fc.demand_estimator.parameters.update(p["demand_estimator"])
    fc.demand_for_goods.parameters.update(p["demand_for_goods"])
    fc.excess_demand.parameters.update(p["excess_demand"])
    fc.target_production.parameters.update(p["target_production"])
    fc.target_capital_inputs.parameters.update(p["target_capital_inputs"])

    country_config.government_entities.functions.consumption.name = p["government_consumption_setter"]
    country_config.households.functions.consumption.name = p["household_consumption_setter"]
    country_config.households.functions.investment.name = p["household_investment_setter"]

    if labour_force_index is None and use_observed_labour_path:
        if province is None or n_quarters is None:
            raise ValueError("use_observed_labour_path=True requires `province` and `n_quarters`.")
        labour_force_index = observed_labour_force_index(
            n_quarters=n_quarters, province=province, uniform=uniform_labour_path
        )

    if labour_force_index is not None:
        country_config.individuals.functions.demography.name = p["demography"]
        country_config.individuals.functions.demography.parameters = {
            "labour_force_index": list(labour_force_index),
            "seed": int(demography_seed),
        }
    return country_config
