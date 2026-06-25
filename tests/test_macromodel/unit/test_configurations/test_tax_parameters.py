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

    def test_unknown_year_raises(self):
        with pytest.raises(KeyError):
            read_tax_parameters("bc", 1900)

    def test_schedule_field_in_file_is_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("bc:\n  2014:\n    pit_brackets: [[1.0, 0.1]]\n")
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

    def test_only_expressible_credits_are_carried(self):
        config = build_central_government_configuration("bc", 2014)
        kinds = {c.kind for c in config.pit_tax_credits}
        # Universal + age-based credits are kept ...
        assert "Personal Amount" in kinds
        assert "Age Amount" in kinds
        # ... household-composition credits are skipped (not yet expressible).
        assert "Spousal Amount" not in kinds
        assert "Equivalent To Spouse Amount" not in kinds

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
