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

    # ── step_pit_brackets: CPI inflation of thresholds ────────────

    def test_step_pit_brackets_inflates_thresholds(
        self, test_central_government_pit,
    ):
        """step_pit_brackets compound-inflates pit_thresholds from
        stored base values."""
        cg = test_central_government_pit

        assert cg.pit_base_thresholds is not None, "PIT base thresholds should be stored"
        original = cg.states["pit_thresholds"].copy()

        # Inflate from base_year=2014 to tax_year=2017 with known CPI
        cpi_map = {2014: 0.01, 2015: 0.02, 2016: 0.015}
        cg.step_pit_brackets(tax_year=2017, cpi_map=cpi_map, base_year=2014)

        factor = 1.01 * 1.02 * 1.015  # ≈ 1.045653
        inflated = cg.states["pit_thresholds"]

        # All thresholds (except inf) should be scaled
        for i in range(len(inflated) - 1):  # last is inf
            expected = original[i] * factor
            assert inflated[i] == pytest.approx(expected), (
                f"threshold[{i}]: {inflated[i]} != {expected}"
            )
        # Last threshold remains inf
        assert np.isinf(inflated[-1])

    def test_step_pit_brackets_noop_at_base_year(
        self, test_central_government_pit,
    ):
        """Call with tax_year <= base_year is a no-op."""
        cg = test_central_government_pit
        original = cg.states["pit_thresholds"].copy()

        cpi_map = {2014: 0.01, 2015: 0.02}
        cg.step_pit_brackets(tax_year=2014, cpi_map=cpi_map, base_year=2014)

        np.testing.assert_array_equal(cg.states["pit_thresholds"], original)

    def test_step_pit_brackets_noop_without_pit(self, test_central_government):
        """Flat-tax government: step_pit_brackets is a no-op."""
        cg = test_central_government
        # Should not raise
        cg.step_pit_brackets(
            tax_year=2017,
            cpi_map={2014: 0.01, 2015: 0.02},
            base_year=2014,
        )

    def test_step_pit_brackets_idempotent(
        self, test_central_government_pit,
    ):
        """Repeated calls with the same arguments give the same result."""
        cg = test_central_government_pit
        cpi_map = {2014: 0.01, 2015: 0.02, 2016: 0.03}

        cg.step_pit_brackets(tax_year=2017, cpi_map=cpi_map, base_year=2014)
        first = cg.states["pit_thresholds"].copy()

        cg.step_pit_brackets(tax_year=2017, cpi_map=cpi_map, base_year=2014)
        second = cg.states["pit_thresholds"].copy()

        np.testing.assert_array_equal(second, first)

    # ── pit_credit_base & pit_taxable_income_deductions ───────

    def test_credit_base_stored(self, test_central_government_pit_full):
        """pit_credit_base is stored in states when configured."""
        cg = test_central_government_pit_full
        assert "pit_credit_base" in cg.states
        assert cg.states["pit_credit_base"] == 0.0

    def test_credit_base_not_stored_without_config(self, test_central_government_pit):
        """Without pit_credit_base in config, state key is absent."""
        assert "pit_credit_base" not in test_central_government_pit.states

    def test_taxable_income_deductions_stored(self, test_central_government_pit_full):
        """pit_taxable_income_deductions stored when configured."""
        cg = test_central_government_pit_full
        assert "pit_taxable_income_deductions" in cg.states
        assert cg.states["pit_taxable_income_deductions"] == 9869.0

    def test_taxable_income_deductions_lower_tax_revenue(
        self, test_central_government_pit_full,
    ):
        """Taxable-income deductions reduce tax revenue vs raw PIT."""
        cg = test_central_government_pit_full

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

        # Tax revenue with deductions < without deductions
        tax_with_deductions = cg.ts.get_aggregate("taxes_income")[-1]

        # Compare: compute raw progressive tax without deductions
        taxable = emp_income * (1 - cg.states["Employee Social Insurance Tax"])
        pit_raw = compute_progressive_tax(
            taxable,
            cg.states["pit_thresholds"],
            cg.states["pit_rates"],
        )
        assert tax_with_deductions < pit_raw.sum(), (
            f"Tax with deductions ({tax_with_deductions:.2f}) should be "
            f"less than raw PIT ({pit_raw.sum():.2f})"
        )

    def test_taxable_income_deductions_are_subtracted_before_brackets(
        self, test_central_government_pit_full,
    ):
        """Taxable-income deductions lower the taxable base before brackets."""
        cg = test_central_government_pit_full

        # Low-income earner whose income is below the deduction threshold
        emp_income = np.array([5000.0])  # less than basic_deduction 9869
        activity = np.array([ActivityStatus.EMPLOYED])

        cg.compute_taxes(
            current_ind_employee_income=emp_income,
            current_total_rent_paid=0.0,
            current_income_financial_assets=np.zeros(1),
            current_ind_activity=activity,
            current_ind_realised_cons=np.zeros(1),
            current_bank_profits=np.zeros(1),
            current_firm_production=np.zeros(1),
            current_firm_price=np.ones(1),
            current_firm_profits=np.zeros(1),
            current_firm_industries=np.zeros(1, dtype=int),
            current_household_new_real_wealth=np.zeros(1),
            taxes_less_subsidies_rates=np.zeros(1),
            current_total_exports=0.0,
        )

        # After deductions, taxable drops to 0 (capped), so tax ≈ 0
        last_tax = cg.ts.get_aggregate("taxes_income")[-1]
        # With employee_SIT removed and deduction > income, tax should be 0
        assert last_tax == pytest.approx(0.0, abs=1e-6), (
            f"Expected ~zero tax for below-deduction income, got {last_tax}"
        )

    # ── step_pit_brackets: deductions are CPI-inflated too ────────

    def test_step_pit_brackets_inflates_credit_base(
        self, test_central_government_pit_full,
    ):
        """CPI inflation inflates pit_credit_base (set to non-zero
        for this test to verify the plumbing)."""
        cg = test_central_government_pit_full
        assert cg.pit_base_credit_base is not None

        # Seed a non-zero value to verify inflation works
        cg.states["pit_credit_base"] = 5000.0
        cg.pit_base_credit_base = 5000.0

        cpi_map = {2014: 0.01, 2015: 0.02, 2016: 0.015}
        cg.step_pit_brackets(tax_year=2017, cpi_map=cpi_map, base_year=2014)

        factor = 1.01 * 1.02 * 1.015
        expected = 5000.0 * factor
        assert cg.states["pit_credit_base"] == pytest.approx(expected)

    def test_step_pit_brackets_inflates_taxable_income_deductions(
        self, test_central_government_pit_full,
    ):
        """CPI inflation also inflates pit_taxable_income_deductions."""
        cg = test_central_government_pit_full
        assert cg.pit_base_deductions is not None

        original = cg.states["pit_taxable_income_deductions"]
        cpi_map = {2014: 0.10}

        cg.step_pit_brackets(tax_year=2015, cpi_map=cpi_map, base_year=2014)

        expected = original * 1.10
        assert cg.states["pit_taxable_income_deductions"] == pytest.approx(expected)

    def test_step_pit_brackets_empty_cpi_noop(
        self, test_central_government_pit_full,
    ):
        """Empty cpi_map: thresholds, deductions, and credit_base unchanged."""
        cg = test_central_government_pit_full
        orig_thresh = cg.states["pit_thresholds"].copy()
        orig_credit = cg.states["pit_credit_base"]
        orig_deduc = cg.states["pit_taxable_income_deductions"]

        cg.step_pit_brackets(tax_year=2017, cpi_map={}, base_year=2014)

        np.testing.assert_array_equal(cg.states["pit_thresholds"], orig_thresh)
        assert cg.states["pit_credit_base"] == orig_credit
        assert cg.states["pit_taxable_income_deductions"] == orig_deduc

    # ── Pre-calibration: effective rate from employee income ─────

    def test_pre_calibration_effective_rate_in_country_construction(
        self, datawrapper, test_individuals,
    ):
        """When country.py constructs a Country with pit_brackets,
        the Income Tax effective rate is pre-calibrated at t=0."""
        from macromodel.agents.central_government import CentralGovernment
        from macromodel.configurations import CentralGovernmentConfiguration

        country_data = datawrapper.synthetic_countries["FRA"]

        pit_config = CentralGovernmentConfiguration(
            pit_brackets=[
                (37606, 0.0506),
                (75213, 0.077),
                (86354, 0.105),
                (104858, 0.1229),
                (150000, 0.147),
                (float("inf"), 0.168),
            ],
        )

        n_unemployed = np.sum(
            test_individuals.states["Activity Status"] == ActivityStatus.UNEMPLOYED
        )

        cg = CentralGovernment.from_pickled_agent(
            synthetic_central_government=country_data.central_government,
            configuration=pit_config,
            country_name="FRA",
            all_country_names=["FRA", "ROW"],
            taxes_net_subsidies=country_data.industry_data["industry_vectors"][
                "Taxes Less Subsidies Rates"
            ].values,
            tax_data=country_data.tax_data,
            n_industries=datawrapper.n_industries,
            number_of_unemployed_individuals=n_unemployed,
        )

        # Effective rate must have been set — not the raw OECD average
        assert "Income Tax" in cg.states
        rate = cg.states["Income Tax"]
        assert 0.05 < rate < 0.17, (
            f"Pre-calibrated effective rate {rate:.4f} should be between 5% and 17%"
        )

    def test_pre_calibration_with_deductions(
        self, test_central_government_pit_full,
    ):
        """With taxable-income deductions, the pre-calibrated effective
        rate is lower than brackets-only."""
        cg = test_central_government_pit_full

        # The effective rate should be lower than the brackets-only case
        # because the taxable-income deduction reduces taxable base
        rate = cg.states["Income Tax"]
        # With BC-like brackets (5.06% lowest), deductions push effective
        # below the no-deduction case
        assert rate < 0.10, (
            f"Effective rate with deductions ({rate:.4f}) should be < 10%"
        )
        assert rate > 0.0
