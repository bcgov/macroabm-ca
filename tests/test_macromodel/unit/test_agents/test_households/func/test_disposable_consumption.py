"""Focused unit tests for DisposableIncomeHouseholdConsumption.

Verifies the disposable-income identity and its consumption consequences directly on the
rule (no full simulation): higher personal taxes cut aggregate consumption, higher transfers
raise it, allocation preserves the aggregate, the smoothing/benefit floors do not create an
upward aggregate-demand ratchet, and the rule is numerically stable at the edges.
"""

import numpy as np

from macromodel.agents.households.func.consumption import (
    DisposableIncomeHouseholdConsumption,
)

N_HH, N_IND = 4, 3
WEIGHTS = np.array([0.5, 0.3, 0.2])


def _rule(smoothing_fraction=0.0, window=4, min_frac=0.0):
    return DisposableIncomeHouseholdConsumption(
        consumption_smoothing_fraction=smoothing_fraction,
        consumption_smoothing_window=window,
        minimum_consumption_fraction=min_frac,
    )


def _call(rule, income, W, F, income_tax, si=0.0, benefits=None, saving=0.3, tau_vat=0.1, history=None):
    income = np.asarray(income, float)
    if benefits is None:
        benefits = np.zeros_like(income)
    if history is None:
        history = np.zeros((2, N_HH))
    return rule.compute_target_consumption(
        expected_inflation=0.0,
        current_cpi=1.0,
        initial_cpi=1.0,
        historic_consumption_sum=np.asarray(history, float),
        saving_rates=np.full(N_HH, saving),
        income=income,
        household_benefits=np.asarray(benefits, float),
        consumption_weights=WEIGHTS,
        consumption_weights_by_income=np.tile(WEIGHTS, (5, 1)),
        exogenous_total_consumption=np.ones(60),
        current_time=1,
        take_consumption_weights_by_income_quantile=False,
        tau_vat=tau_vat,
        income_tax=income_tax,
        employee_social_insurance_tax=si,
        employee_income=np.asarray(W, float),
        financial_income=np.asarray(F, float),
    )


def test_disposable_identity_matches_formula():
    W = np.array([100.0, 80.0, 0.0, 40.0])
    F = np.array([10.0, 0.0, 5.0, 0.0])
    income = W + F + np.array([20.0, 50.0, 30.0, 25.0])  # + transfers/rental
    disp = DisposableIncomeHouseholdConsumption.disposable_income(
        income, W, F, income_tax=0.2, employee_social_insurance_tax=0.1
    )
    expected = income - 0.2 * ((1 - 0.1) * W + F) - 0.1 * W
    assert np.allclose(disp, np.maximum(0.0, expected))


def test_higher_personal_tax_reduces_aggregate_consumption():
    W = np.full(N_HH, 100.0)
    F = np.full(N_HH, 20.0)
    income = W + F + 30.0
    low = _call(_rule(), income, W, F, income_tax=0.10).sum()
    high = _call(_rule(), income, W, F, income_tax=0.30).sum()
    assert high < low


def test_higher_transfers_increase_aggregate_consumption():
    W = np.full(N_HH, 100.0)
    F = np.full(N_HH, 20.0)
    base_income = W + F + 30.0
    more_transfers = base_income + 40.0  # transfers are untaxed, W/F unchanged
    base = _call(_rule(), base_income, W, F, income_tax=0.2, benefits=np.full(N_HH, 30.0)).sum()
    more = _call(_rule(), more_transfers, W, F, income_tax=0.2, benefits=np.full(N_HH, 70.0)).sum()
    assert more > base


def test_allocation_preserves_aggregate_total():
    W = np.full(N_HH, 90.0)
    F = np.full(N_HH, 10.0)
    income = W + F + 25.0
    out = _call(_rule(), income, W, F, income_tax=0.2, tau_vat=0.15)
    per_hh = out.sum(axis=1)
    disp = DisposableIncomeHouseholdConsumption.disposable_income(income, W, F, 0.2, 0.0)
    expected_per_hh = (1 - 0.3) * disp / (1 + 0.15)
    assert np.allclose(per_hh, expected_per_hh)  # weights sum to 1 -> allocation preserves the per-hh total
    assert np.isclose(out.sum(), expected_per_hh.sum())


def test_floors_do_not_create_upward_ratchet():
    # With a smoothing fraction < 1 the smoothing floor is a strict fraction of recent
    # consumption, so a drop in income cannot lock aggregate consumption at the prior high level.
    rule = _rule(smoothing_fraction=0.5, window=4)
    W = np.full(N_HH, 100.0)
    F = np.full(N_HH, 20.0)
    high_income = W + F + 50.0
    hist_high = np.tile(_call(rule, high_income, W, F, income_tax=0.2).sum(axis=1), (4, 1))
    cons_high = _call(rule, high_income, W, F, income_tax=0.2, history=hist_high).sum()
    cons_low = _call(rule, np.zeros(N_HH), np.zeros(N_HH), np.zeros(N_HH), income_tax=0.2, history=hist_high).sum()
    assert cons_low < cons_high  # income drop -> consumption falls (no ratchet up)
    assert cons_low <= 0.5 * cons_high + 1e-6  # floor is a strict fraction of history


def test_numerical_stability_at_edges():
    W = np.array([1e12, 0.0, 50.0, 100.0])
    F = np.array([0.0, 1e11, 0.0, 20.0])
    income = W + F + np.array([0.0, 0.0, 1e9, 30.0])
    for it in (0.0, 0.5, 1.0):
        out = _call(_rule(smoothing_fraction=0.3), income, W, F, income_tax=it, si=0.5)
        assert np.all(np.isfinite(out)) and np.all(out >= 0.0)


def test_falls_back_to_gross_without_components():
    # Missing components -> disposable == gross -> identical to the default rule.
    income = np.array([100.0, 120.0, 80.0, 90.0])
    disp = DisposableIncomeHouseholdConsumption.disposable_income(income, None, None, 0.3, 0.1)
    assert np.allclose(disp, income)
