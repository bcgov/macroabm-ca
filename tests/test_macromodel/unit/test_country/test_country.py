import dataclasses
from pathlib import Path

import numpy as np

from macro_data.readers.taxation import TaxationReader
from macromodel.configurations import (
    CentralGovernmentConfiguration,
    CountryConfiguration,
    ExchangeRatesConfiguration,
)
from macromodel.country import Country
from macromodel.exchange_rates import ExchangeRates

# Committed BC schedules (test fixtures).
#   parents[0]=test_country [1]=unit [2]=test_macromodel [3]=tests [4]=repo root
_COMMITTED_PIT_DIR = (
    Path(__file__).resolve().parents[4]
    / "spoof_data" / "freda" / "personal_income_tax"
)


class TestCountry:
    def test__init(self, datawrapper):
        synthetic_country = datawrapper.synthetic_countries["FRA"]
        country_configuration = CountryConfiguration()

        exchange_rates_config = ExchangeRatesConfiguration()
        exchange_rates_df = datawrapper.exchange_rates
        initial_year = 2014
        country_names = ["FRA"]

        exchange_rates = ExchangeRates.from_data(
            exchange_rates_data=exchange_rates_df,
            exchange_rate_config=exchange_rates_config,
            initial_year=initial_year,
            country_names=country_names,
        )

        emission_factors = np.array(
            [
                datawrapper.emission_factors["coal"],
                datawrapper.emission_factors["gas"],
                datawrapper.emission_factors["oil"],
            ]
        )

        country = Country.from_pickled_country(
            synthetic_country=synthetic_country,
            country_configuration=country_configuration,
            exchange_rates=exchange_rates,
            country_name="FRA",
            all_country_names=["FRA", "ROW"],
            industries=datawrapper.industries,
            initial_year=datawrapper.configuration.year,
            t_max=12,
            running_multiple_countries=False,
            emission_factors_usd=emission_factors,
        )

        assert country is not None

    def test__country(self, test_country):
        assert test_country is not None

    def test_pit_bracket_scaling_does_not_mutate_config(self, datawrapper):
        """Building a Country must not scale the caller's pit_brackets in
        place: repeated construction from the same config stays stable."""
        synthetic_country = datawrapper.synthetic_countries["FRA"]
        scale = synthetic_country.scale
        assert scale > 1, "test data must exercise the bracket-scaling path"

        country_configuration = CountryConfiguration(
            central_government=CentralGovernmentConfiguration(
                pit_brackets=[(50000.0, 0.10), (float("inf"), 0.20)],
            ),
        )
        original = list(country_configuration.central_government.pit_brackets)

        emission_factors = np.array(
            [
                datawrapper.emission_factors["coal"],
                datawrapper.emission_factors["gas"],
                datawrapper.emission_factors["oil"],
            ]
        )

        def build():
            exchange_rates = ExchangeRates.from_data(
                exchange_rates_data=datawrapper.exchange_rates,
                exchange_rate_config=ExchangeRatesConfiguration(),
                initial_year=2014,
                country_names=["FRA"],
            )
            return Country.from_pickled_country(
                synthetic_country=synthetic_country,
                country_configuration=country_configuration,
                exchange_rates=exchange_rates,
                country_name="FRA",
                all_country_names=["FRA", "ROW"],
                industries=datawrapper.industries,
                initial_year=datawrapper.configuration.year,
                t_max=12,
                running_multiple_countries=False,
                emission_factors_usd=emission_factors,
            )

        build()
        country = build()  # second construction from the same config object

        # Caller's config is untouched after repeated construction.
        assert list(country_configuration.central_government.pit_brackets) == original
        # Scale was applied exactly once (not compounded across builds).
        assert country.central_government.states["pit_thresholds"][0] == 50000.0 * scale

    def test_taxation_data_activates_progressive_pit_end_to_end(self, datawrapper):
        """End-to-end: a country carrying taxation data, whose government opts in,
        gets the progressive BC schedule on its central-government agent. This
        exercises the full wired stream: SyntheticCountry.taxation ->
        from_pickled_country -> activate_taxation -> build -> from_pickled_agent."""
        base = datawrapper.synthetic_countries["FRA"]
        scale = base.scale
        # Attach BC taxation data to a *copy* of the synthetic country (leave the
        # shared fixture untouched).
        synthetic_country = dataclasses.replace(
            base, taxation=TaxationReader.from_dir(_COMMITTED_PIT_DIR)
        )

        country_configuration = CountryConfiguration(
            central_government=CentralGovernmentConfiguration(
                activate_progressive_pit=True
            ),
        )

        emission_factors = np.array(
            [
                datawrapper.emission_factors["coal"],
                datawrapper.emission_factors["gas"],
                datawrapper.emission_factors["oil"],
            ]
        )
        exchange_rates = ExchangeRates.from_data(
            exchange_rates_data=datawrapper.exchange_rates,
            exchange_rate_config=ExchangeRatesConfiguration(),
            initial_year=2014,
            country_names=["FRA"],
        )
        country = Country.from_pickled_country(
            synthetic_country=synthetic_country,
            country_configuration=country_configuration,
            exchange_rates=exchange_rates,
            country_name="FRA",
            all_country_names=["FRA", "ROW"],
            industries=datawrapper.industries,
            initial_year=datawrapper.configuration.year,
            t_max=12,
            running_multiple_countries=False,
            emission_factors_usd=emission_factors,
        )

        # Progressive PIT is active: the agent carries the BC schedule, with the
        # first bracket threshold scaled to agent units (37,606 per individual).
        states = country.central_government.states
        assert "pit_thresholds" in states
        assert states["pit_thresholds"][0] == 37606.0 * scale
        assert states["pit_rates"][0] == 0.0506
        # Dividend integration switched on with the schedule present.
        assert states["pit_dividend_integration"] is True

    def test_no_optin_keeps_flat_even_with_taxation_data(self, datawrapper):
        """Taxation data present but the government did not opt in ⇒ no
        progressive PIT (flat parity preserved)."""
        base = datawrapper.synthetic_countries["FRA"]
        synthetic_country = dataclasses.replace(
            base, taxation=TaxationReader.from_dir(_COMMITTED_PIT_DIR)
        )
        country_configuration = CountryConfiguration()  # activate_progressive_pit=False

        emission_factors = np.array(
            [
                datawrapper.emission_factors["coal"],
                datawrapper.emission_factors["gas"],
                datawrapper.emission_factors["oil"],
            ]
        )
        exchange_rates = ExchangeRates.from_data(
            exchange_rates_data=datawrapper.exchange_rates,
            exchange_rate_config=ExchangeRatesConfiguration(),
            initial_year=2014,
            country_names=["FRA"],
        )
        country = Country.from_pickled_country(
            synthetic_country=synthetic_country,
            country_configuration=country_configuration,
            exchange_rates=exchange_rates,
            country_name="FRA",
            all_country_names=["FRA", "ROW"],
            industries=datawrapper.industries,
            initial_year=datawrapper.configuration.year,
            t_max=12,
            running_multiple_countries=False,
            emission_factors_usd=emission_factors,
        )
        assert "pit_thresholds" not in country.central_government.states
