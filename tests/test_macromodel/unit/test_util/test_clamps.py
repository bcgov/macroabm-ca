"""Unit tests for the shared inf-safe input clamp (macromodel.util.clamps)."""

import numpy as np

from macromodel.util.clamps import clamp_towards


class TestClampTowards:
    def test_finite_limit_matches_naive_formula_at_every_weight(self):
        # For finite limits the guarded clamp must equal the naive expression
        # min(target, target + w*(limit - target)) bit-for-bit, at every weight.
        target = np.array([100.0, 120.0, 80.0])
        limit = np.array([90.0, 150.0, 80.0])
        for w in (1.0, 0.75, 0.5, 0.25, 0.0):
            naive = np.minimum(target, target + w * (limit - target))
            assert np.allclose(clamp_towards(target, limit, w), naive)

    def test_infinite_limit_is_no_constraint_at_every_weight(self):
        # An infinite limit means "no binding constraint": target must pass through
        # unchanged for every weight, including w=0.0 where the naive form gives NaN.
        target = np.array([100.0])
        for w in (1.0, 0.5, 0.0):
            out = clamp_towards(target, np.array([np.inf]), w)
            assert out[0] == 100.0

    def test_zero_weight_with_inf_limit_produces_no_nan(self):
        # Regression: 0.0 * inf -> NaN, propagated by np.minimum, then fillna->0
        # silently disabled a firm. The guard must never emit NaN.
        target = np.array([100.0, 100.0])
        limit = np.array([np.inf, 90.0])
        out = clamp_towards(target, limit, 0.0)
        assert not np.isnan(out).any()
        assert out[0] == 100.0 and out[1] == 100.0  # w=0 ignores the limit entirely

    def test_clamps_down_when_limit_binds(self):
        # w=1.0 clamps target down to a finite binding limit.
        out = clamp_towards(np.array([100.0]), np.array([70.0]), 1.0)
        assert out[0] == 70.0
