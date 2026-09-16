"""Unit tests for the settlement of the consumption tax against household wealth.

The tax was booked as government revenue on realised consumption and debited from nobody:
the wealth update read income less rent less goods-market spending, three terms with no tax
among them, so the same amount sat on the revenue line and in household wealth at once.
"""

import numpy as np
import pandas as pd
import pytest

from macromodel.agents.individuals.individual_properties import ActivityStatus

SURPLUS_INCOME = 100_000.0
DEFICIT_INCOME = 10_000.0
RENT = 5_000.0
SPENDING = 40_000.0
CONSUMPTION = 30_000.0
OPENING_WEALTH = 1_000_000.0
TAU_VAT = 0.15


def _no_properties() -> pd.DataFrame:
    """A properties frame owning nothing, so wealth moves only through the financial flow."""
    return pd.DataFrame(
        {
            "Corresponding Owner Household ID": np.array([], dtype=int),
            "Is Owner-Occupied": np.array([], dtype=int),
            "Value": np.array([], dtype=float),
        }
    )


def _set_flows(households, income: float, opening_wealth: float) -> int:
    """Put every household on one set of flows, holding opening_wealth in each financial asset class.

    Returns the number of households the flows were applied to.
    """
    n_households = households.ts.current("n_households")
    n_industries = households.ts.current("nominal_amount_spent_in_lcu").shape[1]
    households.states["Corresponding Inhabited House ID"] = np.full(n_households, -1)

    spending_by_good = np.zeros((n_households, n_industries))
    spending_by_good[:, 0] = SPENDING
    households.ts.override_current("nominal_amount_spent_in_lcu", spending_by_good)
    households.ts.override_current("investment", np.zeros((n_households, n_industries)))
    households.ts.override_current("income", np.full(n_households, income))
    households.ts.override_current("rent", np.full(n_households, RENT))
    households.ts.override_current("consumption", np.full(n_households, CONSUMPTION))

    for held in ("wealth_other_financial_assets", "wealth_deposits"):
        households.ts.override_current(held, np.full(n_households, opening_wealth))

    for zeroed in (
        "wealth_other_real_assets",
        "interest_paid",
        "price_paid_for_property",
        "debt_installments",
        "received_consumption_loans",
        "received_mortgages",
    ):
        households.ts.override_current(zeroed, np.zeros(n_households))

    return n_households


def _financial_assets(households) -> float:
    """Total financial assets held, the quantity the wealth update moves."""
    return households.ts.current("wealth_deposits").sum() + households.ts.current("wealth_other_financial_assets").sum()


def _financial_wealth_change(households, tau_vat: float, income: float = SURPLUS_INCOME) -> float:
    """Run one wealth update over those flows and return the change in financial assets."""
    _set_flows(households, income=income, opening_wealth=OPENING_WEALTH)
    held_before = _financial_assets(households)
    households.update_wealth(housing_data=_no_properties(), tau_cf=0.0, tau_vat=tau_vat)
    return _financial_assets(households) - held_before


def _book_revenue_on(government, consumption: np.ndarray) -> None:
    """Run the tax computation on that consumption with every other base zeroed."""
    government.compute_taxes(
        current_ind_employee_income=np.zeros(1),
        current_total_rent_paid=0.0,
        current_income_financial_assets=np.zeros(1),
        current_ind_activity=np.array([ActivityStatus.EMPLOYED]),
        current_ind_realised_cons=consumption,
        current_bank_profits=np.zeros(1),
        current_firm_production=np.zeros(1),
        current_firm_price=np.zeros(1),
        current_firm_profits=np.zeros(1),
        current_firm_industries=np.zeros(1, dtype=int),
        current_household_new_real_wealth=np.zeros(1),
        taxes_less_subsidies_rates=np.zeros(1),
        current_total_exports=0.0,
    )


class TestConsumptionTaxSettlement:
    def test_an_untaxed_saver_keeps_the_whole_surplus(self, test_households):
        # The behaviour the tax term is added to: at a zero rate the household keeps
        # income less rent less what it spent in the goods market.
        n_households = _set_flows(test_households, income=SURPLUS_INCOME, opening_wealth=OPENING_WEALTH)

        change = _financial_wealth_change(test_households, tau_vat=0.0)

        assert change == pytest.approx(n_households * (SURPLUS_INCOME - RENT - SPENDING))

    def test_an_untaxed_dissaver_draws_down_the_whole_shortfall(self, test_households):
        # The mirror branch, over the same three terms.
        n_households = _set_flows(test_households, income=DEFICIT_INCOME, opening_wealth=OPENING_WEALTH)

        change = _financial_wealth_change(test_households, tau_vat=0.0, income=DEFICIT_INCOME)

        assert change == pytest.approx(n_households * (DEFICIT_INCOME - RENT - SPENDING))

    def test_the_tax_leaves_a_saver(self, test_households):
        untaxed = _financial_wealth_change(test_households, tau_vat=0.0)

        taxed = _financial_wealth_change(test_households, tau_vat=TAU_VAT)

        n_households = test_households.ts.current("n_households")
        assert untaxed - taxed == pytest.approx(TAU_VAT * CONSUMPTION * n_households)

    def test_the_tax_leaves_a_dissaver_too(self, test_households):
        # The same debit on the drawdown branch, which the saver cases never reach.
        untaxed = _financial_wealth_change(test_households, tau_vat=0.0, income=DEFICIT_INCOME)

        taxed = _financial_wealth_change(test_households, tau_vat=TAU_VAT, income=DEFICIT_INCOME)

        n_households = test_households.ts.current("n_households")
        assert untaxed - taxed == pytest.approx(TAU_VAT * CONSUMPTION * n_households)

    def test_the_debit_equals_the_revenue_the_government_books(self, test_households, test_central_government):
        government = test_central_government
        tau_vat = government.states["Value-added Tax"]
        untaxed = _financial_wealth_change(test_households, tau_vat=0.0)
        taxed = _financial_wealth_change(test_households, tau_vat=tau_vat)

        _book_revenue_on(government, test_households.ts.current("consumption"))

        assert untaxed - taxed == pytest.approx(government.ts.current("taxes_vat")[0])
