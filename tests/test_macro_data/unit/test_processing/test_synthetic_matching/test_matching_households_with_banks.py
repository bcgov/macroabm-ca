"""Unit tests for household-to-bank matching algorithms.

Tests the match_households_with_banks_optimal function which uses the
Hungarian algorithm (scipy.optimize.linear_sum_assignment) to optimally
assign households to banks, and the match_households_with_banks_random
fallback.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from macro_data.processing.synthetic_matching.matching_households_with_banks import (
    _MAX_OPTIMAL_HOUSEHOLDS,
    match_households_with_banks_optimal,
    match_households_with_banks_random,
    rescale,
)


# ---------------------------------------------------------------------------
# Minimal fake / mock classes for SyntheticPopulation and SyntheticBanks
#
# Both real classes are ABCs.  We only need household_data (DataFrame) and
# bank_data (DataFrame) + number_of_banks for the matching functions.
# ---------------------------------------------------------------------------


class FakePopulation:
    """Minimal stand-in for SyntheticPopulation used in tests."""

    def __init__(self, household_data: pd.DataFrame):
        self.household_data = household_data


class FakeBanks:
    """Minimal stand-in for SyntheticBanks used in tests."""

    def __init__(self, bank_data: pd.DataFrame):
        self.bank_data = bank_data

    @property
    def number_of_banks(self) -> int:
        return len(self.bank_data)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_population():
    """10 households with varying deposit+debt profiles."""
    rng = np.random.default_rng(42)
    n = 10
    deposits = rng.uniform(1_000, 100_000, n)
    debt = rng.uniform(0, 50_000, n)
    return FakePopulation(
        pd.DataFrame(
            {
                "Wealth in Deposits": deposits,
                "Debt": debt,
                "Corresponding Bank ID": -1,  # unassigned sentinel
            }
        )
    )


@pytest.fixture
def small_banks():
    """3 banks with known balance-sheet sizes."""
    return FakeBanks(
        pd.DataFrame(
            {
                "Deposits from Households": [500_000.0, 200_000.0, 100_000.0],
                "Loans to Households": [300_000.0, 100_000.0, 50_000.0],
            }
        )
    )


@pytest.fixture
def tiny_population():
    """2 households for analytical tests."""
    return FakePopulation(
        pd.DataFrame(
            {
                "Wealth in Deposits": [100.0, 50.0],
                "Debt": [20.0, 10.0],
                "Corresponding Bank ID": -1,
            }
        )
    )


@pytest.fixture
def tiny_banks():
    """2 banks."""
    return FakeBanks(
        pd.DataFrame(
            {
                "Deposits from Households": [200.0, 100.0],
                "Loans to Households": [100.0, 50.0],
            }
        )
    )


@pytest.fixture
def single_bank():
    """A single bank."""
    return FakeBanks(
        pd.DataFrame(
            {
                "Deposits from Households": [500_000.0],
                "Loans to Households": [300_000.0],
            }
        )
    )


# ---------------------------------------------------------------------------
# Tests for rescale()
# ---------------------------------------------------------------------------


class TestRescale:
    def test_rescale__bank_totals_zero__initialized_from_households(
        self, small_population, small_banks
    ):
        """When bank field sum is zero, banks get equal share of household total."""
        small_banks.bank_data["Test Field"] = 0.0
        rescale(small_population, "Wealth in Deposits", small_banks, "Test Field")
        expected_each = small_population.household_data["Wealth in Deposits"].sum() / 3
        np.testing.assert_allclose(
            small_banks.bank_data["Test Field"].values,
            [expected_each] * 3,
        )

    def test_rescale__aligns_household_and_bank_totals(
        self, small_population, small_banks
    ):
        """After rescaling, sum(households[field]) == sum(banks[field])."""
        rescale(
            small_population, "Wealth in Deposits", small_banks, "Deposits from Households"
        )
        hh_total = small_population.household_data["Wealth in Deposits"].sum()
        bank_total = small_banks.bank_data["Deposits from Households"].sum()
        np.testing.assert_allclose(hh_total, bank_total)

    def test_rescale__preserves_bank_proportions(self, small_banks):
        """Rescaling should scale all banks by the same factor."""
        original_ratios = (
            small_banks.bank_data["Deposits from Households"]
            / small_banks.bank_data["Deposits from Households"].sum()
        )
        pop = FakePopulation(
            pd.DataFrame({"Wealth in Deposits": [100_000.0], "Debt": [0.0]})
        )
        rescale(pop, "Wealth in Deposits", small_banks, "Deposits from Households")
        new_ratios = (
            small_banks.bank_data["Deposits from Households"]
            / small_banks.bank_data["Deposits from Households"].sum()
        )
        np.testing.assert_allclose(original_ratios.values, new_ratios.values)


# ---------------------------------------------------------------------------
# Tests for match_households_with_banks_random
# ---------------------------------------------------------------------------


class TestMatchRandom:
    def test_all_households_assigned(self, small_population, small_banks):
        """Every household must have a valid bank ID after random matching."""
        match_households_with_banks_random(small_population, small_banks)
        assigned = small_population.household_data["Corresponding Bank ID"]
        assert len(assigned) == 10
        assert set(assigned.unique()).issubset({0, 1, 2})

    def test_bank_customer_lists_cover_all_households(
        self, small_population, small_banks
    ):
        """Sum of household counts across banks must equal total households."""
        match_households_with_banks_random(small_population, small_banks)
        customer_lists = small_banks.bank_data["Corresponding Households ID"]
        total = sum(len(lst) for lst in customer_lists)
        assert total == 10

    def test_no_household_in_multiple_banks(self, small_population, small_banks):
        """Each household appears in exactly one bank's customer list."""
        match_households_with_banks_random(small_population, small_banks)
        customer_lists = small_banks.bank_data["Corresponding Households ID"]
        all_ids = []
        for lst in customer_lists:
            all_ids.extend(list(lst))
        assert len(all_ids) == len(set(all_ids)) == 10


# ---------------------------------------------------------------------------
# Tests for match_households_with_banks_optimal
# ---------------------------------------------------------------------------


class TestMatchOptimal:
    def test_all_households_assigned(self, small_population, small_banks):
        """Optimal matching must assign every household to a valid bank."""
        match_households_with_banks_optimal(small_population, small_banks)
        assigned = small_population.household_data["Corresponding Bank ID"]
        assert len(assigned) == 10
        assert set(assigned.unique()).issubset({0, 1, 2})

    def test_total_households_match(self, small_population, small_banks):
        """The sum of assigned household counts must equal total households."""
        match_households_with_banks_optimal(small_population, small_banks)
        customer_lists = small_banks.bank_data["Corresponding Households ID"]
        # np.where returns a tuple (indices_array, dtype), extract the array
        total = sum(
            len(lst[0]) if isinstance(lst, tuple) else len(lst)
            for lst in customer_lists
        )
        assert total == 10

    def test_bank_proportions_respected(self, small_population, small_banks):
        """Larger banks should get proportionally more households."""
        match_households_with_banks_optimal(small_population, small_banks)
        bank_sizes = (
            small_banks.bank_data["Deposits from Households"]
            + small_banks.bank_data["Loans to Households"]
        )
        shares = bank_sizes / bank_sizes.sum()
        customer_lists = small_banks.bank_data["Corresponding Households ID"]
        # np.where returns a tuple; extract first element
        counts = np.array([
            len(lst[0]) if isinstance(lst, tuple) else len(lst)
            for lst in customer_lists
        ])
        # Bank 0 has the largest share, so it should get the most households
        assert counts[0] >= counts[1]
        assert counts[0] >= counts[2]
        # Proportional check: counts should be close to shares * 10
        np.testing.assert_allclose(
            counts / counts.sum(), shares.values, atol=0.15
        )

    def test_single_bank__all_assigned(self, small_population, single_bank):
        """With only one bank, every household must be assigned to bank 0."""
        match_households_with_banks_optimal(small_population, single_bank)
        assigned = small_population.household_data["Corresponding Bank ID"]
        assert (assigned == 0).all()

    def test_deterministic_with_seed(self, small_population, small_banks):
        """Same input (with same random seed) should give same assignment.

        The only non-deterministic part is the random remainder allocation
        when floor rounding leaves unassigned slots.
        """
        rng = np.random.default_rng(123)
        deposits = rng.uniform(1_000, 100_000, 3)
        debt = rng.uniform(0, 50_000, 3)
        pop1 = FakePopulation(
            pd.DataFrame(
                {
                    "Wealth in Deposits": deposits.copy(),
                    "Debt": debt.copy(),
                    "Corresponding Bank ID": -1,
                }
            )
        )
        pop2 = FakePopulation(
            pd.DataFrame(
                {
                    "Wealth in Deposits": deposits.copy(),
                    "Debt": debt.copy(),
                    "Corresponding Bank ID": -1,
                }
            )
        )
        banks1 = FakeBanks(
            pd.DataFrame(
                {
                    "Deposits from Households": [500_000.0, 100_000.0],
                    "Loans to Households": [300_000.0, 50_000.0],
                }
            )
        )
        banks2 = FakeBanks(
            pd.DataFrame(
                {
                    "Deposits from Households": [500_000.0, 100_000.0],
                    "Loans to Households": [300_000.0, 50_000.0],
                }
            )
        )

        np.random.seed(42)
        match_households_with_banks_optimal(pop1, banks1)
        np.random.seed(42)
        match_households_with_banks_optimal(pop2, banks2)

        np.testing.assert_array_equal(
            pop1.household_data["Corresponding Bank ID"].values,
            pop2.household_data["Corresponding Bank ID"].values,
        )

    def test_optimal_cost__analytical(self, tiny_population, tiny_banks):
        """For a setup with known optimal, the assignment should be correct.

        Household totals: [120, 60]
        Bank proportions: bank0 share = 300/450 = 2/3 → 2*2/3 ≈ 1.33 → floor 1
                           bank1 share = 150/450 = 1/3 → 2*1/3 ≈ 0.67 → floor 0
        Remainder (1 hh) randomly assigned — we check the key invariant instead.
        """
        match_households_with_banks_optimal(tiny_population, tiny_banks)
        assigned = tiny_population.household_data["Corresponding Bank ID"]
        assert len(assigned) == 2
        assert set(assigned.unique()).issubset({0, 1})

    def test_exceeds_threshold__falls_back_to_random(self):
        """When n_households > _MAX_OPTIMAL_HOUSEHOLDS, warn and use random."""
        n = _MAX_OPTIMAL_HOUSEHOLDS + 1
        large_pop = FakePopulation(
            pd.DataFrame(
                {
                    "Wealth in Deposits": np.full(n, 50_000.0),
                    "Debt": np.full(n, 10_000.0),
                    "Corresponding Bank ID": -1,
                }
            )
        )
        banks = FakeBanks(
            pd.DataFrame(
                {
                    "Deposits from Households": [500_000.0, 100_000.0],
                    "Loans to Households": [300_000.0, 50_000.0],
                }
            )
        )
        with pytest.warns(UserWarning, match="exceeds the optimal-matching"):
            match_households_with_banks_optimal(large_pop, banks)

        # All households should still be assigned
        assigned = large_pop.household_data["Corresponding Bank ID"]
        assert (assigned >= 0).all()
        assert len(assigned) == n

    def test_rescale_called_during_optimal(self, small_population, small_banks):
        """After optimal matching, the rescale invariant should hold."""
        match_households_with_banks_optimal(small_population, small_banks)

        hh_deposits = small_population.household_data["Wealth in Deposits"].sum()
        bank_deposits = small_banks.bank_data["Deposits from Households"].sum()
        np.testing.assert_allclose(hh_deposits, bank_deposits)

        hh_debt = small_population.household_data["Debt"].sum()
        bank_loans = small_banks.bank_data["Loans to Households"].sum()
        np.testing.assert_allclose(hh_debt, bank_loans)

    def test_bank_data_has_corresponding_households(
        self, small_population, small_banks
    ):
        """Each bank's "Corresponding Households ID" column must be populated."""
        match_households_with_banks_optimal(small_population, small_banks)
        for lst in small_banks.bank_data["Corresponding Households ID"]:
            # np.where returns (indices_array, dtype) tuple
            assert isinstance(lst, tuple)
            assert len(lst) > 0
            assert isinstance(lst[0], np.ndarray)
