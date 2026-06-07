"""Unit tests for compute_phi() in src/evaluation.metrics.

All tests are pure function tests — no model, no GPU, no network.
"""

from __future__ import annotations

import pytest

from src.evaluation.metrics import compute_phi


class TestComputePhi:
    """Procedural Fidelity Score (Phi) computation."""

    def test_default_weights(self):
        """Paper formula: 0.5*Coverage + 0.3*StepF1 + 0.2*Kendall."""
        result = compute_phi(0.6, 0.7, 0.8)
        # 0.5*0.6 + 0.3*0.7 + 0.2*0.8 = 0.30 + 0.21 + 0.16 = 0.67
        assert result == pytest.approx(0.67)

    def test_null_coverage_becomes_zero(self):
        """None coverage is coerced to 0.0."""
        result = compute_phi(None, 1.0, 1.0)
        # 0.5*0.0 + 0.3*1.0 + 0.2*1.0 = 0.50
        assert result == pytest.approx(0.50)

    def test_null_step_f1_becomes_zero(self):
        """None StepF1 is coerced to 0.0."""
        result = compute_phi(0.5, None, 0.5)
        # 0.5*0.5 + 0.3*0.0 + 0.2*0.5 = 0.35
        assert result == pytest.approx(0.35)

    def test_null_kendall_becomes_zero(self):
        """None Kendall is coerced to 0.0."""
        result = compute_phi(0.5, 0.5, None)
        # 0.5*0.5 + 0.3*0.5 + 0.2*0.0 = 0.40
        assert result == pytest.approx(0.40)

    def test_all_none(self):
        """All None returns 0.0 (everything coerced)."""
        result = compute_phi(None, None, None)
        assert result == pytest.approx(0.0)

    def test_custom_weight_scheme_040402(self):
        """Phi sensitivity: weights (0.4, 0.4, 0.2)."""
        result = compute_phi(0.6, 0.7, 0.8, w_cov=0.4, w_step=0.4, w_tau=0.2)
        # 0.4*0.6 + 0.4*0.7 + 0.2*0.8 = 0.24 + 0.28 + 0.16 = 0.68
        assert result == pytest.approx(0.68)

    def test_custom_weight_scheme_060202(self):
        """Phi sensitivity: weights (0.6, 0.2, 0.2)."""
        result = compute_phi(0.6, 0.7, 0.8, w_cov=0.6, w_step=0.2, w_tau=0.2)
        # 0.6*0.6 + 0.2*0.7 + 0.2*0.8 = 0.36 + 0.14 + 0.16 = 0.66
        assert result == pytest.approx(0.66)

    def test_rounding_three_decimals(self):
        """Result is rounded to 3 decimal places via round3()."""
        # Choose values that produce a repeating decimal.
        result = compute_phi(1/3, 0.0, 0.0)
        # 0.5 * 0.333... = 0.1666...
        # round3 should give 0.167
        assert result == 0.167

    def test_edge_case_all_zeros(self):
        """All zero inputs produce zero output."""
        result = compute_phi(0.0, 0.0, 0.0)
        assert result == 0.0

    def test_edge_case_perfect_scores(self):
        """All 1.0 produces Phi = 1.0 (regardless of weights)."""
        result = compute_phi(1.0, 1.0, 1.0)
        assert result == 1.0
        # Also with custom weights
        result2 = compute_phi(1.0, 1.0, 1.0, w_cov=0.4, w_step=0.4, w_tau=0.2)
        assert result2 == 1.0
