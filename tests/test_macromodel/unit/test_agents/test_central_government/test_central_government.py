import numpy as np
import pytest

from macro_data.readers.taxation.personal_income_tax.pit_schedule import compute_progressive_tax
from macromodel.agents.individuals.individual_properties import ActivityStatus


class TestCentralGovernment:
    def test__create(self, test_central_government):
        assert test_central_government.country_name == "FRA"

    def test__central_government_states(self, test_central_government):
        assert test_central_government is not None
        for state in [
            "Value-added Tax",
            "Export Tax",
            "Employer Social Insurance Tax",
            "Employee Social Insurance Tax",
            "Profit Tax",
            "Income Tax",
            "Taxes Less Subsidies Rates",
        ]:
            assert state in test_central_government.states.keys()

    def test__central_government_ts(self, test_central_government):
        for ts_key in [
            "unemployment_benefits_by_individual",
            "total_other_benefits",
        ]:
            assert ts_key in test_central_government.ts.get_keys()

    def test__distribute_unemployment_benefits_to_individuals(self, test_central_government):
        benefits = test_central_government.ts.current("unemployment_benefits_by_individual")
        assert np.allclose(
            test_central_government.distribute_unemployment_benefits_to_individuals(
                current_individual_activity_status=np.array([ActivityStatus.EMPLOYED, ActivityStatus.UNEMPLOYED]),
            ),
            np.array([0.0, benefits[0]]),
        )


class TestCentralGovernmentPIT:
    """Progressive PIT: state storage, tax computation, and effective-rate update."""

    def test_pit_thresholds_and_rates_stored(self, test_central_government_pit):
        """PIT brackets are stored as states when pit_brackets is set."""
        cg = test_central_government_pit
        assert "pit_thresholds" in cg.states
        assert "pit_rates" in cg.states
        assert len(cg.states["pit_thresholds"]) == 6
        assert len(cg.states["pit_rates"]) == 6
        # Last threshold is inf
        assert np.isinf(cg.states["pit_thresholds"][-1])

    def test_flat_config_has_no_pit_states(self, test_central_government):
        """Without pit_brackets, pit_thresholds/rates are absent."""
        assert "pit_thresholds" not in test_central_government.states
        assert "pit_rates" not in test_central_government.states

    def test_compute_taxes_progressive_branch(self, test_central_government_pit):
        """Progressive PIT path: tax revenue > 0 and effective rate updated."""
        cg = test_central_government_pit

        # Two employed individuals: low earner (30k) and high earner (100k)
        emp_income = np.array([30000.0, 100000.0])
        activity = np.array([ActivityStatus.EMPLOYED, ActivityStatus.EMPLOYED])

        cg.compute_taxes(
            current_ind_employee_income=emp_income,
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
            current_total_exports=0.0,
        )

        # Income tax revenue should be positive
        last_tax = cg.ts.get_aggregate("taxes_income")[-1]
        assert last_tax > 0, f"Expected positive tax revenue, got {last_tax}"

        # Effective rate should be between the lowest and highest bracket rates
        rate = cg.states["Income Tax"]
        assert 0.05 < rate < 0.17, (
            f"Effective rate {rate:.4f} should be between 5% and 17%"
        )

    def test_compute_taxes_effective_rate_update(self, test_central_government_pit):
        """After compute_taxes, the effective Income Tax rate is
        consistent with the progressive schedule."""
        cg = test_central_government_pit

        emp_income = np.array([50000.0, 50000.0])
        activity = np.array([ActivityStatus.EMPLOYED, ActivityStatus.EMPLOYED])

        cg.compute_taxes(
            current_ind_employee_income=emp_income,
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
            current_total_exports=0.0,
        )

        # Recompute the expected effective rate from the tax paid
        taxable = emp_income * (1 - cg.states["Employee Social Insurance Tax"])
        pit = compute_progressive_tax(
            taxable,
            cg.states["pit_thresholds"],
            cg.states["pit_rates"],
        )
        expected_rate = float(pit.sum() / taxable.sum())

        assert cg.states["Income Tax"] == pytest.approx(expected_rate, rel=1e-10), (
            f"Effective rate {cg.states['Income Tax']} != expected {expected_rate}"
        )

    def test_progressive_vs_flat_tax_ordering(self, test_central_government_pit):
        """Progressive schedule taxes low earners less, high earners more
        than an equivalent flat rate."""
        cg = test_central_government_pit

        low_income = np.array([20000.0])
        high_income = np.array([200000.0])
        thresholds = cg.states["pit_thresholds"]
        rates = cg.states["pit_rates"]

        # The flat effective rate from compute_taxes (approximate)
        # We just check that the progressive tax is lower for low and
        # higher for high compared to the top marginal rate.
        low_tax = compute_progressive_tax(low_income, thresholds, rates)
        high_tax = compute_progressive_tax(high_income, thresholds, rates)

        low_effective = low_tax / low_income
        high_effective = high_tax / high_income

        # Progressive: low earner effective rate < high earner effective rate
        assert low_effective[0] < high_effective[0], (
            f"Low earner rate {low_effective[0]:.4f} should be < "
            f"high earner rate {high_effective[0]:.4f}"
        )
        # High earner effective rate should be > lowest bracket rate
        assert high_effective[0] > rates[0]
        # Low earner effective rate should be < highest bracket rate
        assert low_effective[0] < rates[-1]
