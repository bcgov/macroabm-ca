"""Focused tests for the TFP investment-gating correction.

Confirms that ordinary net capital investment feeds TFP growth ONLY when a real
productivity-investment planner is active. Under the default
``NoProductivityInvestmentPlanner`` a configured ``SimpleTFPGrowth`` base rate is
followed cleanly (capital investment does not alter TFP), while ``NoOpTFPGrowth``
remains exactly inert. The investment-induced channel is preserved as an explicit
option (selecting a real planner re-enables it).

The gating lives in ``Firms.compute_tfp_growth`` / ``Firms._investment_drives_tfp``;
these are exercised on a light stub so no full simulation is built.
"""

from types import SimpleNamespace

import numpy as np

from macromodel.agents.firms.firms import Firms
from macromodel.agents.firms.func.productivity_growth import (
    NoOpTFPGrowth,
    ProductivityGrowth,
    SimpleTFPGrowth,
)
from macromodel.agents.firms.func.productivity_investment_planner import (
    NoProductivityInvestmentPlanner,
    OptimalProductivityInvestmentPlanner,
    SimpleProductivityInvestmentPlanner,
)

N = 3
BASE = 0.0025
ELAST = 0.3


class _FakeTS:
    def __init__(self, executed, production):
        # `executed` is a list of arrays; len() gates the fallback branch, [-1] is "current"
        self.executed_productivity_investment = executed
        self._production = production

    def current(self, name):
        if name == "executed_productivity_investment":
            return self.executed_productivity_investment[-1]
        if name == "production":
            return self._production
        raise KeyError(name)


class _FakeFirm:
    """Minimal object exposing exactly what Firms.compute_tfp_growth touches."""

    # borrow the real methods under test
    compute_tfp_growth = Firms.compute_tfp_growth
    _investment_drives_tfp = Firms._investment_drives_tfp

    def __init__(self, growth_fn, planner, executed=None, production=None, base=BASE, tfp0=None):
        self.functions = {"productivity_growth": growth_fn, "productivity_investment_planner": planner}
        self.states = {"tfp_multiplier": np.ones(N) if tfp0 is None else np.asarray(tfp0, float)}
        self.ts = _FakeTS(
            [] if executed is None else executed,
            np.full(N, 100.0) if production is None else production,
        )
        self.configuration = SimpleNamespace(
            parameters=SimpleNamespace(tfp_base_growth_rate=base, tfp_investment_elasticity=ELAST)
        )

    def compute_productivity_investment(self):  # only hit on the empty-ts fallback
        return np.zeros(N)


def _planner(cls):
    return cls(n_firms=N)


# ---- flag ----------------------------------------------------------------


def test_investment_drives_tfp_flag():
    assert _FakeFirm(SimpleTFPGrowth(), _planner(NoProductivityInvestmentPlanner))._investment_drives_tfp() is False
    assert _FakeFirm(SimpleTFPGrowth(), _planner(SimpleProductivityInvestmentPlanner))._investment_drives_tfp() is True
    assert _FakeFirm(SimpleTFPGrowth(), _planner(OptimalProductivityInvestmentPlanner))._investment_drives_tfp() is True


# ---- NoOp stays exactly inert -------------------------------------------


def test_noop_inert_regardless_of_planner_or_investment():
    big_inv = [np.full(N, 1e9)]
    for planner in (NoProductivityInvestmentPlanner, SimpleProductivityInvestmentPlanner):
        firm = _FakeFirm(NoOpTFPGrowth(), _planner(planner), executed=big_inv)
        assert np.array_equal(firm.compute_tfp_growth(), np.zeros(N))  # bit-for-bit zero


# ---- Simple + planner OFF follows only the base path --------------------


def test_simple_base_only_when_planner_off():
    firm = _FakeFirm(SimpleTFPGrowth(), _planner(NoProductivityInvestmentPlanner))
    assert np.allclose(firm.compute_tfp_growth(), BASE)


def test_capital_investment_does_not_alter_tfp_when_planner_off():
    # Large realized capital investment present in the time series, but the No-op planner
    # is active -> it must NOT enter TFP growth: still exactly the base rate.
    firm_no_inv = _FakeFirm(SimpleTFPGrowth(), _planner(NoProductivityInvestmentPlanner))
    firm_big_inv = _FakeFirm(SimpleTFPGrowth(), _planner(NoProductivityInvestmentPlanner), executed=[np.full(N, 5e8)])
    g0 = firm_no_inv.compute_tfp_growth()
    g1 = firm_big_inv.compute_tfp_growth()
    assert np.allclose(g0, BASE)
    assert np.allclose(g1, BASE)
    assert np.allclose(g0, g1)  # investment made no difference


# ---- investment-induced TFP works when explicitly enabled ---------------


def test_investment_induced_tfp_only_when_planner_on():
    executed = [np.full(N, 5e8)]
    off = _FakeFirm(SimpleTFPGrowth(), _planner(NoProductivityInvestmentPlanner), executed=executed)
    on = _FakeFirm(SimpleTFPGrowth(), _planner(SimpleProductivityInvestmentPlanner), executed=executed)
    g_off = off.compute_tfp_growth()
    g_on = on.compute_tfp_growth()
    assert np.allclose(g_off, BASE)  # gated out
    assert np.all(g_on > BASE + 1e-9)  # investment term active -> above base
    # matches the SimpleTFPGrowth formula: base + eff * (I/Y)^elasticity
    inv, prod = executed[-1], np.full(N, 100.0)
    expected = BASE + SimpleTFPGrowth().investment_effectiveness * (inv / prod) ** ELAST
    assert np.allclose(g_on, expected)


# ---- compounding correctness --------------------------------------------


def test_compounding_matches_geometric():
    tfp = np.ones(N)
    for _ in range(8):
        tfp = ProductivityGrowth.update_tfp(tfp, np.full(N, BASE))
    assert np.allclose(tfp, (1 + BASE) ** 8)


def test_clean_base_compounds_over_updates_with_planner_off():
    # End-to-end at the Firms level: repeatedly applying compute_tfp_growth with the
    # No-op planner compounds purely at the base rate, independent of stored investment.
    firm = _FakeFirm(SimpleTFPGrowth(), _planner(NoProductivityInvestmentPlanner), executed=[np.full(N, 9e8)])
    tfp = np.ones(N)
    for _ in range(10):
        firm.states["tfp_multiplier"] = tfp
        tfp = ProductivityGrowth.update_tfp(tfp, firm.compute_tfp_growth())
    assert np.allclose(tfp, (1 + BASE) ** 10)
