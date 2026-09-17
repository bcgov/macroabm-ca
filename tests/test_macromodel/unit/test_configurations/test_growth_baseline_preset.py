"""Unit tests for the candidate-growth-baseline preset and the bundled observed
labour-force index loader / on-off switch."""

import numpy as np

from macromodel.configurations import CountryConfiguration
from macromodel.configurations.growth_baseline_preset import (
    apply_candidate_growth_baseline,
    observed_labour_force_index,
)


class TestObservedLabourForceIndex:
    def test_index_starts_at_one_and_has_requested_length(self):
        idx = observed_labour_force_index(n_quarters=53, province="CAN_ON")
        assert len(idx) == 53
        assert abs(idx[0] - 1.0) < 1e-9

    def test_bundled_terminal_growth_matches_lfs(self):
        # Bundled data pins the known 2014->2024 provincial labour-force growth.
        on = observed_labour_force_index(n_quarters=54, province="CAN_ON")
        nl = observed_labour_force_index(n_quarters=54, province="CAN_NL")
        assert on[-1] > 1.15  # Ontario ~ +18.8%
        assert nl[-1] < 1.0  # Newfoundland shrinks (~ -0.7%)

    def test_uniform_gives_all_provinces_the_same_path(self):
        d = observed_labour_force_index(n_quarters=40, uniform=True)
        paths = list(d.values())
        assert all(np.allclose(paths[0], p) for p in paths)


class TestApplyPreset:
    def test_default_preserves_legacy_demography(self):
        c = apply_candidate_growth_baseline(CountryConfiguration.n_industry_default(n_industries=43))
        assert c.individuals.functions.demography.name == "NoAging"
        assert c.firms.functions.demand_for_goods.parameters["unmet_demand_weight"] == 0.25
        assert c.firms.functions.target_capital_inputs.parameters["rolling_reference"] is True

    def test_flag_loads_bundled_labour_path(self):
        c = apply_candidate_growth_baseline(
            CountryConfiguration.n_industry_default(n_industries=43),
            use_observed_labour_path=True,
            province="CAN_AB",
            n_quarters=53,
        )
        assert c.individuals.functions.demography.name == "ExogenousLabourForcePath"
        idx = c.individuals.functions.demography.parameters["labour_force_index"]
        assert len(idx) == 53 and abs(idx[0] - 1.0) < 1e-9

    def test_flag_requires_province_and_quarters(self):
        import pytest

        with pytest.raises(ValueError):
            apply_candidate_growth_baseline(
                CountryConfiguration.n_industry_default(n_industries=43), use_observed_labour_path=True
            )

    def test_explicit_index_overrides_flag(self):
        c = apply_candidate_growth_baseline(
            CountryConfiguration.n_industry_default(n_industries=43), labour_force_index=[1.0, 1.01, 1.02]
        )
        assert c.individuals.functions.demography.parameters["labour_force_index"] == [1.0, 1.01, 1.02]
