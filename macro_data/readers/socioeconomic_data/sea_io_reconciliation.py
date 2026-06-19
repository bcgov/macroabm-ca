"""Province-aware reconciliation between WIOD SEA and provincial IO accounts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from macro_data.configuration.countries import Country
from macro_data.configuration.region import Region


DEFAULT_ACTIVE_VA_FLOOR_ANNUAL = 1e4
DEFAULT_ENFORCE_ACTIVE_VA_ELIGIBILITY = True
CAPITAL_ALLOCATION_ELIGIBLE_FIELD = "Capital Allocation Eligible"


def reconcile_provincial_sea_to_io(
    sea: pd.DataFrame,
    value_added_dict: dict[str | Country | Region, pd.Series],
    regions_dict: dict[Country, list[Region]] | None,
    active_va_floor: float = DEFAULT_ACTIVE_VA_FLOOR_ANNUAL,
    enforce_active_va_eligibility: bool = DEFAULT_ENFORCE_ACTIVE_VA_ELIGIBILITY,
) -> pd.DataFrame:
    """Reconcile province-sector SEA fields to IO value-added constraints.

    This keeps IO value added as the hard province-sector target, while using
    national SEA fields as composition totals to distribute labour compensation,
    capital compensation, and capital stock across active provincial cells.

    The key rule is province-aware eligibility: cells with effectively zero IO
    value added do not receive capital compensation or capital stock. This
    prevents national SEA capital structure from being pushed into sparse
    province-sector cells before the investment matrix is built.
    """

    if regions_dict is None:
        return _reconcile_country_level_sea(sea, value_added_dict)

    reconciled = sea.copy()
    reconciled[CAPITAL_ALLOCATION_ELIGIBLE_FIELD] = 1.0
    for country, regions in regions_dict.items():
        if country not in value_added_dict:
            continue
        region_list = list(regions)
        sectors = value_added_dict[country].index

        for sector in sectors:
            regional_va = pd.Series(
                {
                    region: float(value_added_dict[region].reindex([sector]).fillna(0.0).iloc[0])
                    for region in region_list
                }
            )
            active = regional_va > active_va_floor if enforce_active_va_eligibility else regional_va >= 0.0
            all_index = pd.MultiIndex.from_product([region_list, [sector]], names=reconciled.index.names)
            reconciled.loc[all_index, CAPITAL_ALLOCATION_ELIGIBLE_FIELD] = active.astype(float).to_numpy()

            if not active.any():
                reconciled.loc[
                    all_index,
                    ["Value Added", "Labour Compensation", "Capital Compensation", "Capital Stock"],
                ] = 0.0
                continue

            active_regions = regional_va.index[active]
            active_index = pd.MultiIndex.from_product([active_regions, [sector]], names=reconciled.index.names)
            inactive_regions = regional_va.index[~active]
            inactive_index = pd.MultiIndex.from_product([inactive_regions, [sector]], names=reconciled.index.names)
            national_values = _national_sector_values(reconciled, region_list, sector)

            # Value added is the provincial IO hard constraint.
            reconciled.loc[all_index, "Value Added"] = regional_va.reindex(region_list).to_numpy(dtype=float)

            if len(inactive_index) > 0:
                reconciled.loc[
                    inactive_index,
                    ["Labour Compensation", "Capital Compensation", "Capital Stock"],
                ] = 0.0

            labour_total = min(national_values["Labour Compensation"], regional_va[active].sum())
            capital_total = min(
                national_values["Capital Compensation"],
                max(0.0, regional_va[active].sum() - labour_total),
            )
            stock_total = national_values["Capital Stock"]

            labour_weights = _safe_weights(regional_va[active])
            capital_weights = _safe_weights(regional_va[active])
            stock_weights = _safe_weights(regional_va[active])

            reconciled.loc[active_index, "Labour Compensation"] = labour_total * labour_weights
            reconciled.loc[active_index, "Capital Compensation"] = capital_total * capital_weights
            reconciled.loc[active_index, "Capital Stock"] = stock_total * stock_weights

    return reconciled.fillna(0.0).sort_index()


def get_capital_allocation_eligibility(
    sea: pd.DataFrame,
    country_name: str | Country | Region,
    industries: list[str],
) -> np.ndarray:
    """Return the reconciled active-cell mask used by investment allocation."""

    if CAPITAL_ALLOCATION_ELIGIBLE_FIELD not in sea.columns:
        return np.ones(len(industries), dtype=bool)

    index = pd.MultiIndex.from_product([[country_name], industries], names=sea.index.names)
    eligibility = sea.reindex(index)[CAPITAL_ALLOCATION_ELIGIBLE_FIELD].fillna(1.0).to_numpy(dtype=float)
    return eligibility > 0.5


def _reconcile_country_level_sea(
    sea: pd.DataFrame,
    value_added_dict: dict[str | Country | Region, pd.Series],
) -> pd.DataFrame:
    reconciled = sea.copy()
    for country, value_added in value_added_dict.items():
        if country not in reconciled.index.get_level_values(0):
            continue
        country_index = pd.IndexSlice[country, value_added.index]
        reconciled.loc[country_index, "Value Added"] = value_added.to_numpy(dtype=float)
        va = value_added.to_numpy(dtype=float)
        cap = reconciled.loc[country_index, "Capital Compensation"].to_numpy(dtype=float)
        lab = reconciled.loc[country_index, "Labour Compensation"].to_numpy(dtype=float)
        scale = np.divide(va, lab + cap, out=np.ones_like(va), where=(lab + cap) > va)
        reconciled.loc[country_index, "Labour Compensation"] = lab * scale
        reconciled.loc[country_index, "Capital Compensation"] = cap * scale
    return reconciled.fillna(0.0).sort_index()


def _national_sector_values(sea: pd.DataFrame, regions: list[Region], sector: str) -> pd.Series:
    index = pd.MultiIndex.from_product([regions, [sector]], names=sea.index.names)
    return sea.reindex(index).fillna(0.0).sum(axis=0)


def _safe_weights(values: pd.Series) -> np.ndarray:
    weights = values.to_numpy(dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if weights.sum() == 0.0:
        return np.repeat(1.0 / len(weights), len(weights))
    return weights / weights.sum()
