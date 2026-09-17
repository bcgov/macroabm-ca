"""Unit tests for the opt-in firm growth mechanisms:
- demand smoothing (alpha) in DefaultDemandEstimator
- unmet-demand weighting (rho) in DefaultDemandSetter
- rolling capital reference in FinancialTargetCapitalInputsSetter
- inf-safe input clamps in the excess-demand and target-production setters

Each test also pins the DEFAULT (legacy) behaviour so backward compatibility is
guarded.
"""

import numpy as np

from macromodel.agents.firms.func.demand_estimator import DefaultDemandEstimator
from macromodel.agents.firms.func.demand_for_goods import DefaultDemandSetter
from macromodel.agents.firms.func.excess_demand import ConstrainedExcessDemandSetter
from macromodel.agents.firms.func.target_capital_inputs import FinancialTargetCapitalInputsSetter


class TestUnmetDemandWeight:
    def test_default_rho_is_legacy_sum(self):
        # Default rho=1.0 reproduces demand = sales + excess exactly.
        s = DefaultDemandSetter()
        out = s.compute_demand(sell_real=np.array([10.0, 5.0]), excess_demand=np.array([2.0, 4.0]))
        assert np.allclose(out, [12.0, 9.0])

    def test_rho_scales_only_the_unmet_term(self):
        s = DefaultDemandSetter(unmet_demand_weight=0.25)
        out = s.compute_demand(sell_real=np.array([10.0, 5.0]), excess_demand=np.array([2.0, 4.0]))
        assert np.allclose(out, [10.0 + 0.25 * 2.0, 5.0 + 0.25 * 4.0])


class TestDemandSmoothing:
    def test_default_alpha_is_legacy_no_smoothing(self):
        # demand_smoothing defaults to 1.0 -> smoothed demand == previous demand,
        # so the estimate is the historical multiplicative rule exactly.
        e = DefaultDemandEstimator(sectoral_growth_adjustment_speed=1.0, firm_growth_adjustment_speed=1.0)
        prev = np.array([100.0, 50.0])
        out = e.compute_estimated_demand(
            previous_demand=prev, current_estimated_growth=0.02, estimated_growth_by_firm=np.array([0.0, 0.0])
        )
        assert np.allclose(out, prev * 1.02)

    def test_alpha_smooths_across_calls(self):
        # With alpha<1 the smoothed base is a partial-adjustment blend of successive
        # observations; first call seeds on the first observation.
        e = DefaultDemandEstimator(
            sectoral_growth_adjustment_speed=0.0, firm_growth_adjustment_speed=0.0, demand_smoothing=0.5
        )
        zero = np.array([0.0])
        # speeds=0 so estimate == smoothed demand; feed 100 then 200.
        first = e.compute_estimated_demand(
            previous_demand=np.array([100.0]), current_estimated_growth=0.0, estimated_growth_by_firm=zero
        )
        second = e.compute_estimated_demand(
            previous_demand=np.array([200.0]), current_estimated_growth=0.0, estimated_growth_by_firm=zero
        )
        assert np.allclose(first, [100.0])  # seeds on first obs
        assert np.allclose(second, [0.5 * 100.0 + 0.5 * 200.0])  # partial adjustment


class TestRollingCapitalReference:
    def _setter(self, rolling):
        return FinancialTargetCapitalInputsSetter(
            target_capital_inputs_fraction=0.1, credit_gap_fraction=0.0, rolling_reference=rolling
        )

    def test_default_is_base_year_reference(self):
        # rolling_reference defaults False -> K_ref = (Q_prev/Q_init) * K_init.
        s = self._setter(rolling=False)
        ref = s._reference_capital_stock(
            unconstrained_target_production=None,
            prev_production=np.array([110.0]),
            prev_capital_inputs_stock=np.array([[50.0]]),
            initial_capital_inputs_stock=np.array([[40.0]]),
            initial_production=np.array([100.0]),
        )
        assert np.allclose(ref, [[1.10 * 40.0]])

    def test_rolling_reference_scales_prev_stock_by_desired_growth(self):
        # rolling: K_ref = K_{t-1} * (Q_desired / Q_{t-1}).
        s = self._setter(rolling=True)
        ref = s._reference_capital_stock(
            unconstrained_target_production=np.array([110.0]),
            prev_production=np.array([100.0]),
            prev_capital_inputs_stock=np.array([[50.0]]),
            initial_capital_inputs_stock=np.array([[40.0]]),
            initial_production=np.array([100.0]),
        )
        assert np.allclose(ref, [[50.0 * 1.10]])

    def test_rolling_safeguards_zero_prev_production_and_zero_prev_capital(self):
        s = self._setter(rolling=True)
        # firm 0: Q_prev=0 -> ratio->1 (hold); firm 1: K_prev=0 but K_init>0 -> base-year fallback (>0)
        ref = s._reference_capital_stock(
            unconstrained_target_production=np.array([110.0, 110.0]),
            prev_production=np.array([0.0, 100.0]),
            prev_capital_inputs_stock=np.array([[50.0], [0.0]]),
            initial_capital_inputs_stock=np.array([[40.0], [40.0]]),
            initial_production=np.array([100.0, 100.0]),
        )
        assert not np.isnan(ref).any()
        assert ref[0, 0] == 50.0  # zero prev-production -> hold
        assert np.isclose(ref[1, 0], 1.10 * 40.0)  # zero prev-capital -> base-year fallback (recovery)


class TestExcessDemandInfSafe:
    def test_default_weights_no_nan_with_infinite_limit(self):
        # Shipped default (capital weight 1.0) with limCap=inf must not emit NaN.
        s = ConstrainedExcessDemandSetter(
            consider_intermediate_inputs=0.0, consider_capital_inputs=1.0, consider_labour_inputs=0.0
        )
        out = s.set_maximum_excess_demand(
            current_production=np.array([100.0, 100.0]),
            target_production=np.array([120.0, 120.0]),
            limiting_intermediate_inputs=np.array([200.0, 200.0]),
            limiting_capital_inputs=np.array([np.inf, 105.0]),
            limiting_labour_inputs=np.array([np.inf, np.inf]),
        )
        assert not np.isnan(out).any()
        assert out[0] == 120.0  # inf limCap -> uncapped -> target
        assert out[1] == 5.0  # limCap-production = 105-100 = 5

    def test_zero_capital_weight_with_inf_limit_no_nan(self):
        # consider_capital_inputs=0.0 is the setting a fix would use; must not NaN.
        s = ConstrainedExcessDemandSetter(
            consider_intermediate_inputs=0.0, consider_capital_inputs=0.0, consider_labour_inputs=0.0
        )
        out = s.set_maximum_excess_demand(
            current_production=np.array([100.0]),
            target_production=np.array([120.0]),
            limiting_intermediate_inputs=np.array([200.0]),
            limiting_capital_inputs=np.array([np.inf]),
            limiting_labour_inputs=np.array([np.inf]),
        )
        assert not np.isnan(out).any()
        assert out[0] == 120.0
