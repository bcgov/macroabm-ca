"""Unit tests for the get_histogram degenerate-distribution robustness fix.
Previously np.histogram raised 'Too many bins for data range' on all-equal or
non-finite inputs, crashing the whole simulation from a diagnostic call."""

import numpy as np

from macromodel.util.get_histogram import fillna, get_histogram


class TestGetHistogramRobustness:
    def test_all_equal_values_do_not_raise(self):
        # Degenerate (zero-range) distribution: must return a well-formed histogram,
        # not raise "Too many bins for data range".
        out = get_histogram(np.full(50, 3.0), scale=None, bins=40)
        assert out.shape == (2, 41)
        assert np.isfinite(out[0][np.isfinite(out[0])]).all()

    def test_non_finite_values_are_tolerated(self):
        vals = np.array([1.0, 2.0, np.inf, np.nan, -np.inf, 3.0])
        out = get_histogram(vals, scale=None, bins=10)
        assert out.shape == (2, 11)

    def test_normal_distribution_still_works(self):
        rng = np.random.default_rng(0)
        out = get_histogram(rng.normal(size=500), scale=None, bins=40)
        assert out.shape == (2, 41)

    def test_fillna_replaces_nan_with_zero_by_default(self):
        assert np.allclose(fillna(np.array([1.0, np.nan, 3.0])), [1.0, 0.0, 3.0])
