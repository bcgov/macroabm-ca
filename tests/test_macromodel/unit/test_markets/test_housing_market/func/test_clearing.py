"""Unit tests for housing market clearing algorithms.

Tests the AutomaticHousingMarketClearer which uses the Hungarian algorithm
(scipy.optimize.linear_sum_assignment) for optimal matching of households
to properties in both sales and rental markets.
"""

import numpy as np
import pandas as pd
import pytest

from macromodel.markets.housing_market.func.clearing import (
    AutomaticHousingMarketClearer,
    DefaultHousingMarketClearer,
    NoHousingMarketClearer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clearer_no_shock():
    """Automatic clearer with zero random shock for deterministic tests."""
    return AutomaticHousingMarketClearer(random_assignment_shock_variance=0.0)


@pytest.fixture
def clearer_with_shock():
    """Automatic clearer with non-zero random shock."""
    return AutomaticHousingMarketClearer(random_assignment_shock_variance=0.5)


@pytest.fixture
def default_clearer():
    """Default (sequential) clearer for comparison tests."""
    return DefaultHousingMarketClearer(random_assignment_shock_variance=0.0)


@pytest.fixture
def small_housing_data():
    """Housing DataFrame with 4 properties: 2 for sale, 1 for rent, 1 occupied.

    Property 0: sale at 200k, owner=10
    Property 1: sale at 300k, owner=11
    Property 2: rent at 1k/mo, owner=12
    Property 3: occupied (not available for either market), owner=13
    """
    return pd.DataFrame(
        {
            "House ID": [0, 1, 2, 3],
            "Value": [200_000, 300_000, 250_000, 180_000],
            "Rent": [1_500, 2_000, 1_000, 800],
            "Sale Price": [200_000, 300_000, 250_000, 180_000],
            "Temporarily for Sale": [True, True, False, False],
            "Up for Rent": [False, False, True, False],
            "Corresponding Inhabitant Household ID": [0, 1, 2, 3],
            "Corresponding Owner Household ID": [10, 11, 12, 13],
            "Is Owner-Occupied": [True, False, True, True],
        }
    )


# ---------------------------------------------------------------------------
# AutomaticHousingMarketClearer -- perform_matching (sales)
# ---------------------------------------------------------------------------


class TestPerformMatchingSales:
    """Tests for perform_matching with is_rental_market=False."""

    def test_all_affordable__matches_every_household(
        self, clearer_no_shock, small_housing_data
    ):
        """When all households have positive WTP, every one is matched."""
        wtp = np.array([250_000, 350_000, 0, 0])  # hh 0,1 have demand
        housing_copy = small_housing_data.copy()
        result = clearer_no_shock.perform_matching(
            housing_data=housing_copy,
            household_main_residence_tenure_status=np.zeros(4),
            max_willing_to_pay=wtp,
            is_rental_market=False,
        )
        assert len(result) == 2
        assert set(result["buyer_id"]) == {0, 1}
        assert all(result["sales_types"] == "Sell")
        # Properties marked unavailable after matching
        unavailable = housing_copy.loc[
            result["property_id"].values, "Temporarily for Sale"
        ]
        assert not unavailable.any()

    def test_low_wtp__still_matched_to_closest_property(
        self, clearer_no_shock, small_housing_data
    ):
        """Households are matched to the property whose price is closest to their WTP.

        Even when WTP is below all prices (negative surplus), the Hungarian
        minimises |WTP - price| so the household is still assigned to the
        cheapest / closest property.
        """
        wtp = np.array([50_000, 350_000, 0, 0])
        result = clearer_no_shock.perform_matching(
            housing_data=small_housing_data.copy(),
            household_main_residence_tenure_status=np.zeros(4),
            max_willing_to_pay=wtp,
            is_rental_market=False,
        )
        # Both households with positive WTP get matched — hh0 to the cheaper property
        assert len(result) == 2
        hh0_match = result[result["buyer_id"] == 0]
        # hh0 should get the cheaper property (200k vs 300k)
        assert hh0_match["price_or_rent"].iloc[0] == 200_000

    def test_no_demand__returns_empty(self, clearer_no_shock):
        """When no household has positive WTP, return an empty DataFrame."""
        housing_data = pd.DataFrame(
            {
                "House ID": [0],
                "Value": [100_000],
                "Rent": [1_000],
                "Sale Price": [100_000],
                "Temporarily for Sale": [True],
                "Up for Rent": [False],
                "Corresponding Inhabitant Household ID": [0],
                "Corresponding Owner Household ID": [10],
                "Is Owner-Occupied": [True],
            }
        )
        wtp = np.array([0.0])
        result = clearer_no_shock.perform_matching(
            housing_data=housing_data,
            household_main_residence_tenure_status=np.zeros(1),
            max_willing_to_pay=wtp,
            is_rental_market=False,
        )
        assert len(result) == 0
        assert list(result.columns) == [
            "sales_types",
            "property_id",
            "property_value",
            "price_or_rent",
            "seller_id",
            "buyer_id",
        ]

    def test_no_available_properties__returns_empty(self, clearer_no_shock):
        """When no properties are listed for sale, return an empty DataFrame."""
        housing_data = pd.DataFrame(
            {
                "House ID": [0],
                "Value": [100_000],
                "Rent": [1_000],
                "Sale Price": [100_000],
                "Temporarily for Sale": [False],
                "Up for Rent": [False],
                "Corresponding Inhabitant Household ID": [0],
                "Corresponding Owner Household ID": [10],
                "Is Owner-Occupied": [True],
            }
        )
        wtp = np.array([200_000])
        result = clearer_no_shock.perform_matching(
            housing_data=housing_data,
            household_main_residence_tenure_status=np.zeros(1),
            max_willing_to_pay=wtp,
            is_rental_market=False,
        )
        assert len(result) == 0

    def test_more_households_than_properties__some_unmatched(
        self, clearer_no_shock
    ):
        """With excess demand, only as many matches as available properties."""
        housing_data = pd.DataFrame(
            {
                "House ID": [0, 1],
                "Value": [100_000, 150_000],
                "Rent": [1_000, 1_200],
                "Sale Price": [100_000, 150_000],
                "Temporarily for Sale": [True, True],
                "Up for Rent": [False, False],
                "Corresponding Inhabitant Household ID": [0, 1],
                "Corresponding Owner Household ID": [10, 11],
                "Is Owner-Occupied": [True, False],
            }
        )
        wtp = np.array([200_000, 300_000, 250_000])  # 3 hh, 2 properties
        result = clearer_no_shock.perform_matching(
            housing_data=housing_data.copy(),
            household_main_residence_tenure_status=np.zeros(3),
            max_willing_to_pay=wtp,
            is_rental_market=False,
        )
        assert len(result) == 2
        # Each property matched at most once
        assert len(set(result["property_id"])) == 2

    def test_more_properties_than_households__excess_remain(
        self, clearer_no_shock, small_housing_data
    ):
        """With excess supply, only as many matches as households with demand."""
        wtp = np.array([300_000, 0, 0, 0])  # only hh 0 has demand
        result = clearer_no_shock.perform_matching(
            housing_data=small_housing_data.copy(),
            household_main_residence_tenure_status=np.zeros(4),
            max_willing_to_pay=wtp,
            is_rental_market=False,
        )
        assert len(result) == 1
        assert result["buyer_id"].iloc[0] == 0

    def test_transaction_dataframe_schema(
        self, clearer_no_shock, small_housing_data
    ):
        """The output DataFrame must have the exact expected columns and dtypes."""
        wtp = np.array([250_000, 350_000, 0, 0])
        result = clearer_no_shock.perform_matching(
            housing_data=small_housing_data.copy(),
            household_main_residence_tenure_status=np.zeros(4),
            max_willing_to_pay=wtp,
            is_rental_market=False,
        )
        expected_columns = [
            "sales_types",
            "property_id",
            "property_value",
            "price_or_rent",
            "seller_id",
            "buyer_id",
        ]
        assert list(result.columns) == expected_columns
        assert result["property_id"].dtype in (np.int64, np.int32, int)
        assert result["seller_id"].dtype in (np.int64, np.int32, int)
        assert result["buyer_id"].dtype in (np.int64, np.int32, int)

    def test_optimal_assignment__known_solution(
        self, clearer_no_shock
    ):
        """Smoke test: verify the Hungarian picks the globally optimal match.

        Two households (WTP: 300k and 200k) and two properties (priced 290k and 150k).
        Optimal: hh0→property0 (cost=10k), hh1→property1 (cost=50k) = total 60k.
        The alternative (hh0→1 cost=150k, hh1→0 cost=90k = 240k) is worse.
        """
        housing_data = pd.DataFrame(
            {
                "House ID": [0, 1],
                "Value": [290_000, 150_000],
                "Rent": [1_500, 1_000],
                "Sale Price": [290_000, 150_000],
                "Temporarily for Sale": [True, True],
                "Up for Rent": [False, False],
                "Corresponding Inhabitant Household ID": [0, 1],
                "Corresponding Owner Household ID": [10, 11],
                "Is Owner-Occupied": [True, False],
            }
        )
        wtp = np.array([300_000, 200_000])  # hh0 has 300k, hh1 has 200k
        result = clearer_no_shock.perform_matching(
            housing_data=housing_data.copy(),
            household_main_residence_tenure_status=np.zeros(2),
            max_willing_to_pay=wtp,
            is_rental_market=False,
        )
        # Build mapping from buyer_id → property_id
        mapping = dict(zip(result["buyer_id"], result["property_id"]))
        # hh0 should get the 290k house (property 0), hh1 gets the 150k house
        assert mapping[0] == 0  # hh0 gets more expensive property
        assert mapping[1] == 1

    def test_households_already_operated__excluded(
        self, clearer_no_shock, small_housing_data
    ):
        """Households passed via households_already_operated must not be matched."""
        wtp = np.array([250_000, 350_000, 0, 0])
        result = clearer_no_shock.perform_matching(
            housing_data=small_housing_data.copy(),
            household_main_residence_tenure_status=np.zeros(4),
            max_willing_to_pay=wtp,
            is_rental_market=False,
            households_already_operated=np.array([0]),  # exclude hh 0
        )
        assert 0 not in set(result["buyer_id"])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# AutomaticHousingMarketClearer -- perform_matching (rental)
# ---------------------------------------------------------------------------


class TestPerformMatchingRental:
    """Tests for perform_matching with is_rental_market=True."""

    def test_rental_matching_uses_rent_prices(
        self, clearer_no_shock, small_housing_data
    ):
        """Rental matching should use "Rent"/"Up for Rent" not sale columns."""
        wtp = np.array([0, 0, 1_200, 0])  # hh 2 wants rental
        result = clearer_no_shock.perform_matching(
            housing_data=small_housing_data.copy(),
            household_main_residence_tenure_status=np.zeros(4),
            max_willing_to_pay=wtp,
            is_rental_market=True,
        )
        assert len(result) == 1
        assert all(result["sales_types"] == "Rental")
        # Should get property 2 (the only one up for rent)
        assert result["property_id"].iloc[0] == 2
        assert result["price_or_rent"].iloc[0] == 1_000

    def test_rental_low_wtp__still_matched_to_closest_rent(
        self, clearer_no_shock, small_housing_data
    ):
        """Household matched to the closest rent even when WTP is below rent.

        The Hungarian minimises |WTP - rent|, so the household still
        gets the closest rental property.
        """
        wtp = np.array([0, 0, 500, 0])  # hh 2 can pay 500, rent is 1000
        result = clearer_no_shock.perform_matching(
            housing_data=small_housing_data.copy(),
            household_main_residence_tenure_status=np.zeros(4),
            max_willing_to_pay=wtp,
            is_rental_market=True,
        )
        # Matched to the only available rental (property 2, rent=1000)
        assert len(result) == 1
        assert result["buyer_id"].iloc[0] == 2
        assert result["price_or_rent"].iloc[0] == 1_000


# ---------------------------------------------------------------------------
# AutomaticHousingMarketClearer -- clear (combined)
# ---------------------------------------------------------------------------


class TestClear:
    """Tests for the full clear() method (sales + rental combined)."""

    def test_clear__combines_sales_and_rental(
        self, clearer_no_shock, small_housing_data
    ):
        """clear() returns both sales and rental transactions."""
        price_wtp = np.array([250_000, 350_000, 0, 0])
        rent_wtp = np.array([0, 0, 1_200, 0])
        result = clearer_no_shock.clear(
            housing_data=small_housing_data.copy(),
            household_main_residence_tenure_status=np.zeros(4),
            max_price_willing_to_pay=price_wtp,
            max_rent_willing_to_pay=rent_wtp,
        )
        assert {"Sell", "Rental"}.issubset(set(result["sales_types"]))
        assert len(result) == 3  # 2 sales + 1 rental

    def test_clear__no_household_in_both_markets(
        self, clearer_no_shock, small_housing_data
    ):
        """A household matched in sales must not appear in rental results."""
        # hh 0 can both buy and rent — should only appear in sales
        price_wtp = np.array([250_000, 0, 1_200, 0])
        rent_wtp = np.array([250_000, 0, 1_200, 0])
        result = clearer_no_shock.clear(
            housing_data=small_housing_data.copy(),
            household_main_residence_tenure_status=np.zeros(4),
            max_price_willing_to_pay=price_wtp,
            max_rent_willing_to_pay=rent_wtp,
        )
        # hh 0 should appear at most once
        buyer_counts = result["buyer_id"].value_counts()
        assert buyer_counts.get(0, 0) <= 1

    def test_clear__integer_ids(self, clearer_no_shock, small_housing_data):
        """The combined output should have integer property/seller/buyer IDs."""
        price_wtp = np.array([250_000, 350_000, 0, 0])
        rent_wtp = np.array([0, 0, 1_200, 0])
        result = clearer_no_shock.clear(
            housing_data=small_housing_data.copy(),
            household_main_residence_tenure_status=np.zeros(4),
            max_price_willing_to_pay=price_wtp,
            max_rent_willing_to_pay=rent_wtp,
        )
        assert result["property_id"].dtype in (np.int64, np.int32, int)
        assert result["seller_id"].dtype in (np.int64, np.int32, int)
        assert result["buyer_id"].dtype in (np.int64, np.int32, int)

    def test_clear__no_demand_returns_empty(self, clearer_no_shock, small_housing_data):
        """When no household wants to buy or rent, return empty."""
        result = clearer_no_shock.clear(
            housing_data=small_housing_data.copy(),
            household_main_residence_tenure_status=np.zeros(4),
            max_price_willing_to_pay=np.zeros(4),
            max_rent_willing_to_pay=np.zeros(4),
        )
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Random shock variance tests
# ---------------------------------------------------------------------------


class TestRandomShockVariance:
    """Tests for the effect of random_assignment_shock_variance."""

    def test_shock_can_change_assignment(self, small_housing_data):
        """With a large shock variance, assignments may differ from deterministic.

        We run multiple trials; at least one should differ from the no-shock
        assignment when the shock is large enough and there are ties to break.
        """
        # Build a scenario with near-identical prices so shocks can flip assignments
        housing_data = pd.DataFrame(
            {
                "House ID": [0, 1],
                "Value": [100_000, 100_000],
                "Rent": [1_000, 2_000],
                "Sale Price": [100_000, 100_000],
                "Temporarily for Sale": [True, True],
                "Up for Rent": [False, False],
                "Corresponding Inhabitant Household ID": [0, 1],
                "Corresponding Owner Household ID": [10, 11],
                "Is Owner-Occupied": [True, False],
            }
        )
        wtp = np.array([200_000, 200_000])

        # Deterministic run (multiple times to get baseline)
        clearer_zero = AutomaticHousingMarketClearer(
            random_assignment_shock_variance=0.0
        )
        baseline_results = []
        for _ in range(50):
            result = clearer_zero.perform_matching(
                housing_data=housing_data.copy(),
                household_main_residence_tenure_status=np.zeros(2),
                max_willing_to_pay=wtp,
                is_rental_market=False,
            )
            baseline_results.append(
                tuple(result.sort_values("buyer_id")["property_id"])
            )

        # With shock — should differ occasionally
        clearer_high = AutomaticHousingMarketClearer(
            random_assignment_shock_variance=0.5
        )
        shock_results = []
        for _ in range(50):
            result = clearer_high.perform_matching(
                housing_data=housing_data.copy(),
                household_main_residence_tenure_status=np.zeros(2),
                max_willing_to_pay=wtp,
                is_rental_market=False,
            )
            shock_results.append(
                tuple(result.sort_values("buyer_id")["property_id"])
            )

        # At least one high-shock run should differ from deterministic baseline
        unique_baseline = set(baseline_results)
        unique_shock = set(shock_results)
        assert len(unique_shock) > 1 or unique_shock != unique_baseline, (
            f"Expected shock to sometimes alter assignment, "
            f"baseline: {unique_baseline}, shocked: {unique_shock}"
        )

    def test_zero_shock_deterministic(self, small_housing_data):
        """With zero variance, the same input always produces the same result."""
        wtp = np.array([300_000, 200_000, 0, 0])
        first_result = None
        for _ in range(10):
            clearer = AutomaticHousingMarketClearer(
                random_assignment_shock_variance=0.0
            )
            result = clearer.perform_matching(
                housing_data=small_housing_data.copy(),
                household_main_residence_tenure_status=np.zeros(4),
                max_willing_to_pay=wtp,
                is_rental_market=False,
            )
            current = tuple(
                result.sort_values("buyer_id")["property_id"]
            )
            if first_result is None:
                first_result = current
            else:
                assert current == first_result, "Zero-shock matching was non-deterministic"


# ---------------------------------------------------------------------------
# Other clearer types (baseline sanity)
# ---------------------------------------------------------------------------


class TestOtherClearers:
    """Sanity tests for NoHousingMarketClearer and DefaultHousingMarketClearer."""

    def test_no_clearer_returns_empty(self, small_housing_data):
        """The null clearer always returns an empty DataFrame."""
        clearer = NoHousingMarketClearer(random_assignment_shock_variance=0.0)
        result = clearer.clear(
            housing_data=small_housing_data,
            household_main_residence_tenure_status=np.zeros(4),
            max_price_willing_to_pay=np.array([1, 2, 3, 4]),
            max_rent_willing_to_pay=np.array([1, 2, 3, 4]),
        )
        assert len(result) == 0

    def test_default_clearer_no_duplicate_buyers(self, small_housing_data):
        """Default sequential clearer should never assign same buyer twice."""
        clearer = DefaultHousingMarketClearer(
            random_assignment_shock_variance=0.0
        )
        # hh 0 can afford both sales and rental
        result = clearer.clear(
            housing_data=small_housing_data.copy(),
            household_main_residence_tenure_status=np.zeros(4),
            max_price_willing_to_pay=np.array([300_000, 0, 0, 0]),
            max_rent_willing_to_pay=np.array([300_000, 0, 0, 0]),
        )
        buyer_counts = result["buyer_id"].value_counts()
        assert buyer_counts.get(0, 0) <= 1
