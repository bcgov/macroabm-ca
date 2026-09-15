"""Tests for the construction-time GDP identity checks in `create_economy_timeseries`."""

import numpy as np
import pytest

from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.economy.economy_ts import create_economy_timeseries

# A balanced economy: output, expenditure and income each reconcile to 87.0.
#   output      = 100 - 30 + 5 - 2 + 10 + 4
#   expenditure = 1 + 6 + 50 + 20 + 8 - 12 + 10 + 4
#   income      = 30 + 40 + 5 + 8 + 0 + 0 + 4
BALANCED_GDP = 87.0

# The constructor addresses individual sectors by position and slices as far as [17:], so it
# needs at least eighteen. Twenty keeps the per-sector values exact: 20 x 5.0 = 100.0 sales
# against 20 x 1.5 = 30.0 intermediate consumption.
N_SECTORS = 20
SECTORAL_SALES = np.full(N_SECTORS, 5.0)
SECTORAL_USED_II = np.full(N_SECTORS, 1.5)


def balanced_kwargs() -> dict:
    """Return arguments for an economy whose output, expenditure and income legs agree."""
    return {
        "country_name": "AAA",
        "all_country_names": ["AAA"],
        "n_industries": len(SECTORAL_SALES),
        "initial_firm_prices": np.ones(len(SECTORAL_SALES)),
        "initial_firm_total_sales": SECTORAL_SALES.sum(),
        "initial_sectoral_firm_sales": SECTORAL_SALES.copy(),
        "initial_sectoral_firm_used_ii": SECTORAL_USED_II.copy(),
        "initial_total_taxes_on_products": 5.0,
        "initial_total_taxes_on_production": 2.0,
        "initial_change_in_firm_stock_inventories": 1.0,
        "initial_gross_fixed_capital_formation": 6.0,
        "initial_total_operating_surplus": 30.0,
        "initial_total_wages": 40.0,
        "initial_individual_activity": np.array(
            [ActivityStatus.EMPLOYED, ActivityStatus.EMPLOYED, ActivityStatus.UNEMPLOYED]
        ),
        "initial_cpi_inflation": 0.0,
        "initial_ppi_inflation": 0.0,
        "initial_hpi_inflation": 0.0,
        "initial_real_rent_paid": np.array([10.0]),
        "initial_imp_rent_paid": np.array([4.0]),
        "initial_hh_rental_income": np.array([8.0]),
        "initial_hh_consumption": 50.0,
        "initial_gov_consumption": 20.0,
        "initial_cg_rent_received": 0.0,
        "initial_cg_taxes_rental_income": 0.0,
        "initial_imports": np.array([12.0]),
        "initial_imports_by_country": {},
        "initial_exports": np.array([8.0]),
        "initial_exports_by_country": {},
        "export_taxes": 0.0,
        "initial_total_growth": 0.0,
        "initial_npl_ratio": 0.0,
    }


class TestGDPIdentityChecks:
    def test__income_leg_mismatch_is_rejected(self):
        """An income-only discrepancy raises.

        This is the case the second assert existed for and did not cover: it duplicated the
        expenditure comparison, so output/income was never enforced. Expenditure is left
        reconciling here, so only the income check can reject this economy.
        """
        kwargs = balanced_kwargs()
        kwargs["initial_total_wages"] += 1.0

        with pytest.raises(AssertionError, match="output/income"):
            create_economy_timeseries(**kwargs)
