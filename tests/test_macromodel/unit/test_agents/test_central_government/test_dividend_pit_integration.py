"""Integration tests for the bank-dividend → PIT pipeline.

Covers three layers:
  1. Pool arithmetic — ``build_dividend_tax_items`` with ``small_business_share``
     variations, and how the grossed-up result enters ``build_taxable_income_pool``.
  2. End-to-end revenue — ``CentralGovernment.compute_taxes`` with a bank
     dividend present raises PIT revenue and matches the manual schedule.
  3. States propagation — ``bank_dividend_small_business_share`` flows from
     ``CentralGovernmentConfiguration`` into ``CentralGovernment.states``.
"""

import numpy as np
import pytest

from macromodel.agents.central_government.pit_pools import (
    PitContext,
    build_dividend_tax_items,
    build_taxable_income_pool,
)
from macromodel.agents.individuals.individual_properties import ActivityStatus

# 2014 BC rates used across all tests.
_ELIG_GU = 0.38
_NONELIG_GU = 0.18
_ELIG_DTC = 0.10
_NONELIG_DTC = 0.0259


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bank_items(dividend_income, small_business_share):
    return build_dividend_tax_items(
        dividend_income=np.asarray(dividend_income, dtype=float),
        small_business_share=small_business_share,
        eligible_gross_up=_ELIG_GU,
        non_eligible_gross_up=_NONELIG_GU,
        eligible_dtc_rate=_ELIG_DTC,
        non_eligible_dtc_rate=_NONELIG_DTC,
    )


# ---------------------------------------------------------------------------
# 1. Pool-level arithmetic
# ---------------------------------------------------------------------------

class TestBankDividendPool:
    """build_dividend_tax_items + PitContext for the bank path (pure functions)."""

    def test_all_eligible_when_share_zero(self):
        """s=0 (banks' definitional case) yields 100% eligible treatment."""
        grossed, dtc = _bank_items([100.0], 0.0)
        # 100 × 1.38 = 138; DTC = 0.10 × 138 = 13.8
        np.testing.assert_allclose(grossed, [138.0])
        np.testing.assert_allclose(dtc, [13.8])

    @pytest.mark.parametrize("s", [0.0, 0.5, 1.0])
    def test_small_business_share_hand_calc(self, s):
        """Gross-up and DTC match manual split for any share value."""
        D = 1000.0
        grossed, dtc = _bank_items([D], s)

        elig = (1.0 - s) * D
        non_elig = s * D
        expected_grossed = elig * (1 + _ELIG_GU) + non_elig * (1 + _NONELIG_GU)
        expected_dtc = _ELIG_DTC * elig * (1 + _ELIG_GU) + _NONELIG_DTC * non_elig * (1 + _NONELIG_GU)

        np.testing.assert_allclose(grossed, [expected_grossed], rtol=1e-9)
        np.testing.assert_allclose(dtc, [expected_dtc], rtol=1e-9)

    def test_share_zero_dominates_share_one(self):
        """s=0 yields higher gross-up and DTC than s=1 for the same dividend."""
        D = np.array([1000.0])
        grossed_0, dtc_0 = _bank_items(D, 0.0)
        grossed_1, dtc_1 = _bank_items(D, 1.0)
        assert grossed_0.item() > grossed_1.item()
        assert dtc_0.item() > dtc_1.item()

    def test_grossed_up_bank_dividend_stacks_into_pool(self):
        """Grossed-up bank dividend adds to Pool A exactly like other streams."""
        grossed, _ = _bank_items([100.0], 0.0)  # 138

        ctx = PitContext(
            employee_income=np.array([1000.0]),
            employee_si_rate=0.0,
            grossed_up_dividend=grossed,
        )
        pool = build_taxable_income_pool(ctx)
        np.testing.assert_allclose(pool, [1138.0])

    def test_zero_dividend_contributes_nothing(self):
        """Non-investors (dividend_income=0) leave the pool unchanged."""
        grossed, dtc = _bank_items([0.0, 100.0], 0.0)

        np.testing.assert_allclose(grossed[0], 0.0)
        np.testing.assert_allclose(dtc[0], 0.0)


# ---------------------------------------------------------------------------
# 2. End-to-end revenue
# ---------------------------------------------------------------------------

class TestBankDividendPITRevenue:
    """compute_taxes with grossed-up bank dividend raises PIT and matches schedule."""

    _EMP_INCOME = np.array([50_000.0, 80_000.0])
    _ACTIVITY = np.array([ActivityStatus.EMPLOYED, ActivityStatus.EMPLOYED])

    def _run(self, cg, grossed_up=None, dtc=None):
        cg.compute_taxes(
            current_ind_employee_income=self._EMP_INCOME,
            current_total_rent_paid=0.0,
            current_income_financial_assets=np.zeros(2),
            current_ind_activity=self._ACTIVITY,
            current_ind_realised_cons=np.zeros(2),
            current_bank_profits=np.zeros(1),
            current_firm_production=np.zeros(1),
            current_firm_price=np.ones(1),
            current_firm_profits=np.zeros(1),
            current_firm_industries=np.zeros(1, dtype=int),
            current_household_new_real_wealth=np.zeros(1),
            taxes_less_subsidies_rates=np.zeros(1),
            grossed_up_dividend_per_ind=grossed_up,
            direct_credits_per_ind=dtc,
        )
        return cg.ts.get_aggregate("taxes_income")[-1]

    def test_bank_dividend_increases_pit_revenue(self, test_central_government_pit):
        """PIT revenue is strictly higher when a bank dividend enters the pool.

        Individuals are placed in the top BC bracket (16.8%) so the extra PIT
        from the grossed-up dividend (16.8% × 1.38 × D = 0.232 × D) always
        exceeds the eligible DTC (10% × 1.38 × D = 0.138 × D).
        """
        cg = test_central_government_pit
        # Use income large enough to sit in the 16.8% top bracket regardless
        # of the synthetic-population SI deduction rate.
        high_income = np.array([500_000.0, 600_000.0])
        activity = np.array([ActivityStatus.EMPLOYED, ActivityStatus.EMPLOYED])

        def _run_high(grossed_up=None, dtc=None):
            cg.compute_taxes(
                current_ind_employee_income=high_income,
                current_total_rent_paid=0.0,
                current_income_financial_assets=np.zeros(2),
                current_ind_activity=activity,
                current_ind_realised_cons=np.zeros(2),
                current_bank_profits=np.zeros(1),
                current_firm_production=np.zeros(1),
                current_firm_price=np.ones(1),
                current_firm_profits=np.zeros(1),
                current_firm_industries=np.zeros(1, dtype=int),
                current_household_new_real_wealth=np.zeros(1),
                taxes_less_subsidies_rates=np.zeros(1),
                grossed_up_dividend_per_ind=grossed_up,
                direct_credits_per_ind=dtc,
            )
            return cg.ts.get_aggregate("taxes_income")[-1]

        bank_div = np.array([5_000.0, 10_000.0])
        grossed, dtc = _bank_items(bank_div, 0.0)

        rev_baseline = _run_high()
        rev_with_div = _run_high(grossed_up=grossed, dtc=dtc)

        assert rev_with_div > rev_baseline, (
            f"PIT revenue did not increase: {rev_with_div} vs {rev_baseline}"
        )

    def test_bank_dividend_pit_matches_hand_calc(self, test_central_government_pit):
        """PIT revenue from bank dividend matches the manual schedule application."""
        from macro_data.readers.taxation.personal_income_tax.pit_schedule import (
            compute_progressive_tax,
        )

        cg = test_central_government_pit
        si_rate = float(cg.states["Employee Social Insurance Tax"])
        bank_div = np.array([5_000.0, 10_000.0])
        grossed, dtc = _bank_items(bank_div, 0.0)

        # Pool A: wages net of SI + grossed-up bank dividend
        taxable = self._EMP_INCOME * (1 - si_rate) + grossed
        pit_gross = compute_progressive_tax(
            taxable, cg.states["pit_thresholds"], cg.states["pit_rates"]
        )
        # No credit_base credits in the test fixture; only the direct DTC (2b)
        expected = float(np.maximum(0.0, pit_gross - dtc).sum())

        actual = self._run(cg, grossed_up=grossed, dtc=dtc)
        assert actual == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# 3. States propagation
# ---------------------------------------------------------------------------

class TestBankDividendStatePropagate:
    """bank_dividend_small_business_share flows from config into CG states."""

    def test_default_share_zero_in_states(self, test_central_government_pit):
        """Default (0.0) is present in states after from_pickled_agent."""
        cg = test_central_government_pit
        assert "bank_dividend_small_business_share" in cg.states
        assert cg.states["bank_dividend_small_business_share"] == pytest.approx(0.0)

    def test_explicit_share_propagates(self, datawrapper):
        """A non-default share set in config reaches states unchanged."""
        from macromodel.agents.central_government import CentralGovernment
        from macromodel.configurations import CentralGovernmentConfiguration

        country = datawrapper.synthetic_countries["FRA"]
        taxes_ls = country.industry_data["industry_vectors"]["Taxes Less Subsidies Rates"].values

        config = CentralGovernmentConfiguration(
            pit_brackets=[(float("inf"), 0.10)],
            bank_dividend_small_business_share=0.25,
        )
        cg = CentralGovernment.from_pickled_agent(
            synthetic_central_government=country.central_government,
            configuration=config,
            country_name="FRA",
            all_country_names=["FRA", "ROW"],
            taxes_net_subsidies=taxes_ls,
            tax_data=country.tax_data,
            n_industries=datawrapper.n_industries,
            number_of_unemployed_individuals=0,
        )
        assert cg.states["bank_dividend_small_business_share"] == pytest.approx(0.25)
