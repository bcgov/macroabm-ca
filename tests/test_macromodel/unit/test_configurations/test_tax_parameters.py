"""Tests for the BC/Canada tax scalar-parameter reader and config builder."""

import math

import pytest

from macromodel.configurations import (
    CentralGovernmentConfiguration,
    apply_tax_parameters,
    build_central_government_configuration,
    read_tax_parameters,
)
from macromodel.configurations.tax_parameters.tax_parameters_reader import (
    _ALLOWED_FIELDS,
)


class TestReadTaxParameters:
    def test_returns_only_scalar_fields(self):
        overrides = read_tax_parameters("bc", 2014)
        assert set(overrides).issubset(_ALLOWED_FIELDS)
        # Schedule fields must never appear in the scalar file.
        assert "pit_brackets" not in overrides
        assert "pit_tax_credits" not in overrides

    def test_values_match_packaged_yaml(self):
        overrides = read_tax_parameters("bc", 2014)
        assert overrides["dividend_small_business_share"] == 0.90
        assert overrides["couple_rental_income_split"] == 0.5
        assert overrides["pit_dividend_integration"] is False

    def test_unknown_jurisdiction_raises(self):
        with pytest.raises(KeyError):
            read_tax_parameters("atlantis", 2014)

    def test_year_before_all_available_raises(self):
        # 1900 precedes every available block (2014+), so there is no prior
        # year to fall back to -> KeyError.
        with pytest.raises(KeyError):
            read_tax_parameters("bc", 1900)

    def test_absent_later_year_falls_back_to_latest_prior(self):
        # 2015 is not packaged (only 2014); its scalar assumptions fall back to
        # the latest prior year (2014) rather than raising.
        assert read_tax_parameters("bc", 2015) == read_tax_parameters("bc", 2014)

    def test_schedule_field_in_file_is_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("bc:\n  2014:\n    pit_brackets: [[1.0, 0.1]]\n")
        with pytest.raises(ValueError, match="Schedule field"):
            read_tax_parameters("bc", 2014, path=bad)

    def test_dividend_rate_field_in_file_is_rejected(self, tmp_path):
        """Dividend gross-up / DTC rates are schedule-sourced; the YAML rejects them."""
        bad = tmp_path / "bad.yaml"
        bad.write_text("bc:\n  2014:\n    dividend_eligible_gross_up: 0.38\n")
        with pytest.raises(ValueError, match="Schedule field"):
            read_tax_parameters("bc", 2014, path=bad)

    def test_unknown_field_in_file_is_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("bc:\n  2014:\n    made_up_knob: 1.0\n")
        with pytest.raises(ValueError, match="Unrecognised"):
            read_tax_parameters("bc", 2014, path=bad)


class TestApplyTaxParameters:
    def test_overrides_scalars_preserves_schedules(self):
        base = CentralGovernmentConfiguration(
            pit_brackets=[(50000.0, 0.1), (math.inf, 0.2)],
            couple_rental_income_split=0.9,
        )
        applied = apply_tax_parameters(base, "bc", 2014)
        # Scalar overridden from the YAML ...
        assert applied.couple_rental_income_split == 0.5
        # ... while the schedule the caller set is left untouched.
        assert applied.pit_brackets == [(50000.0, 0.1), (math.inf, 0.2)]

    def test_parity_with_defaults(self):
        """The packaged values mirror the config defaults, so applying them to a
        fresh config must be a no-op (behaviour stays at parity until opted in)."""
        base = CentralGovernmentConfiguration()
        applied = apply_tax_parameters(base, "bc", 2014)
        assert applied.model_dump() == base.model_dump()


class TestBuildCentralGovernmentConfiguration:
    def test_brackets_from_schedule(self):
        config = build_central_government_configuration("bc", 2014)
        assert config.pit_brackets[0] == (37606.0, 0.0506)
        # Top bracket is open-ended.
        assert math.isinf(config.pit_brackets[-1][0])
        assert config.pit_brackets[-1][1] == 0.168

    def test_only_runtime_applicable_credits_are_carried(self):
        config = build_central_government_configuration("bc", 2014)
        kinds = {c.kind for c in config.pit_tax_credits}
        # Universal, age-based, and the household-composition credits the runtime
        # dispatches by kind (Spousal, Equivalent-To-Spouse) are carried.
        assert "Personal Amount" in kinds
        assert "Age Amount" in kinds
        assert "Spousal Amount" in kinds
        assert "Equivalent To Spouse Amount" in kinds
        # Per-child amounts stay deferred (runtime branch exists but needs
        # child-count context).
        assert "Child Amount Under 18" not in kinds
        assert "Child Amount Under 6" not in kinds
        # Pension Income Amount is NOT age-based (CPP may start 60-70 and the
        # credit covers non-CPP pension income); without pension-income data it
        # is deferred rather than proxied by age, so it must not be carried.
        assert "Pension Income Amount" not in kinds

    def test_household_credits_apply_through_build_to_run(self):
        """End-to-end: the Spousal and Equivalent-To-Spouse credits carried by
        the builder actually reduce tax through the runtime credit pool.  Age no
        longer gates them; the runtime dispatches by kind and tests household
        composition.  The contribution is isolated by differencing the same pool
        with and without the two credits."""
        import numpy as np

        from macromodel.agents.central_government.pit_pools import (
            PitContext,
            build_credit_base_pool,
        )
        from macromodel.agents.households.household_properties import HouseholdType

        config = build_central_government_configuration("bc", 2014)
        spousal_amt = next(
            c.amount for c in config.pit_tax_credits if c.kind == "Spousal Amount"
        )
        equiv_amt = next(
            c.amount
            for c in config.pit_tax_credits
            if c.kind == "Equivalent To Spouse Amount"
        )

        # Convert the built credits into the runtime states-dict form (mirrors
        # CentralGovernment.from_synthetic).
        def to_defs(credits):
            return [
                {
                    "kind": t.kind,
                    "amount": t.amount,
                    "indexing": t.indexing,
                    "age_min": t.eligibility_age_min,
                    "clawback_start": t.clawback_start,
                    "clawback_cap": t.clawback_cap,
                }
                for t in credits
            ]

        # Individuals 0,1 are a couple (household 0); individual 2 is a single
        # parent (household 1).  Individual 1 has zero income, so individual 0
        # receives the full Spousal Amount.
        taxable = np.array([60000.0, 0.0, 40000.0])
        ctx = PitContext(
            employee_income=taxable,
            employee_si_rate=0.0,
            individuals_age=np.array([40, 40, 40]),
            individuals_corr_households=np.array([0, 0, 1]),
            households_type=np.array(
                [
                    HouseholdType.TWO_ADULTS_YOUNGER_THAN_65,
                    HouseholdType.SINGLE_PARENT_WITH_CHILDREN,
                ]
            ),
            households_n_adults=np.array([2, 1]),
        )

        all_defs = to_defs(config.pit_tax_credits)
        wo_defs = [
            d
            for d in all_defs
            if d["kind"] not in {"Spousal Amount", "Equivalent To Spouse Amount"}
        ]
        delta = build_credit_base_pool(all_defs, taxable, ctx) - build_credit_base_pool(
            wo_defs, taxable, ctx
        )

        # Individual 0: spouse (individual 1) has zero income -> full Spousal.
        assert delta[0] == pytest.approx(spousal_amt)
        # Individual 1: spouse (individual 0) earns 60k > amount -> Spousal zero.
        assert delta[1] == pytest.approx(0.0)
        # Individual 2: single parent -> Equivalent-To-Spouse amount.
        assert delta[2] == pytest.approx(equiv_amt)

    def test_age_credit_carries_age_and_clawback(self):
        config = build_central_government_configuration("bc", 2014)
        age = next(c for c in config.pit_tax_credits if c.kind == "Age Amount")
        assert age.eligibility_age_min == 65
        assert age.clawback_start == 32943.0
        assert age.clawback_cap == 62450.0

    def test_scalars_applied(self):
        config = build_central_government_configuration("bc", 2014)
        assert config.dividend_small_business_share == 0.90
        assert config.couple_rental_income_split == 0.5

    def test_dividend_rates_from_schedule(self):
        """The gross-up / DTC rates come from the dividend schedule CSV."""
        config = build_central_government_configuration("bc", 2014)
        assert config.dividend_eligible_gross_up == pytest.approx(0.38)
        assert config.dividend_non_eligible_gross_up == pytest.approx(0.18)
        assert config.dividend_eligible_dtc_rate == pytest.approx(0.10)
        assert config.dividend_non_eligible_dtc_rate == pytest.approx(0.0259)

    def test_dividend_integration_activates_when_schedule_present(self):
        """Presence of the dividend schedule CSV switches integration on
        automatically, overriding the (False) YAML fallback switch."""
        config = build_central_government_configuration("bc", 2014)
        assert config.pit_dividend_integration is True

    def test_dividend_integration_off_when_schedule_absent(self):
        """With no dividend schedule the builder leaves integration off (legacy
        treatment) instead of failing; the bracket schedule still builds."""
        config = build_central_government_configuration(
            "bc", 2014, dividend_schedule_filename="does_not_exist.csv"
        )
        assert config.pit_dividend_integration is False
        # The rest of the configuration is unaffected.
        assert config.pit_brackets[0] == (37606.0, 0.0506)

    def test_brackets_not_pre_scaled(self):
        """Country.from_pickled_country applies the agent-scale; the builder must
        return per-individual units, not pre-scaled ones."""
        config = build_central_government_configuration("bc", 2014)
        # 37,606 is the per-individual first threshold, not 37,606 * scale.
        assert config.pit_brackets[0][0] == 37606.0

    def test_does_not_mutate_default_path(self):
        """Building the BC config must not change a fresh default config."""
        default = CentralGovernmentConfiguration()
        build_central_government_configuration("bc", 2014)
        assert default.pit_brackets is None

    def test_later_year_builds_with_scalar_fallback(self):
        """A later year the YAML lacks still builds: brackets CPI-index, the
        dividend rates come from the year-ranged CSV (distinct 2019 values),
        and the scalar assumptions fall back to the latest prior year (2014)."""
        config = build_central_government_configuration("bc", 2019)
        # 2019 dividend rates differ from 2014 (the CSV is year-ranged).
        assert config.dividend_eligible_dtc_rate == pytest.approx(0.12)
        assert config.dividend_non_eligible_gross_up == pytest.approx(0.15)
        # Scalar assumptions fall back to the 2014 block.
        assert config.dividend_small_business_share == 0.90
        assert config.couple_rental_income_split == 0.5
