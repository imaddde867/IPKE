"""Unit tests for compute_phi() in src/evaluation.metrics.

All tests are pure function tests — no model, no GPU, no network.
"""

from __future__ import annotations

import pytest

from src.evaluation.metrics import compute_phi


class TestComputePhi:
    """Procedural Fidelity Score (Phi) computation."""

    def test_default_weights_unrounded(self):
        """Paper formula: 0.5*Coverage + 0.3*StepF1 + 0.2*Kendall (default, unrounded)."""
        result = compute_phi(0.6, 0.7, 0.8)
        expected = 0.5 * 0.6 + 0.3 * 0.7 + 0.2 * 0.8
        assert result == pytest.approx(expected)

    def test_null_coverage_becomes_zero(self):
        """None coverage is coerced to 0.0."""
        result = compute_phi(None, 1.0, 1.0)
        assert result == pytest.approx(0.5)

    def test_null_step_f1_becomes_zero(self):
        """None StepF1 is coerced to 0.0."""
        result = compute_phi(0.5, None, 0.5)
        assert result == pytest.approx(0.35)

    def test_null_kendall_becomes_zero(self):
        """None Kendall is coerced to 0.0."""
        result = compute_phi(0.5, 0.5, None)
        assert result == pytest.approx(0.40)

    def test_all_none(self):
        """All None returns 0.0 (everything coerced)."""
        result = compute_phi(None, None, None)
        assert result == pytest.approx(0.0)

    def test_custom_weight_scheme_040402(self):
        """Phi sensitivity: weights (0.4, 0.4, 0.2)."""
        result = compute_phi(0.6, 0.7, 0.8, w_cov=0.4, w_step=0.4, w_tau=0.2)
        expected = 0.4 * 0.6 + 0.4 * 0.7 + 0.2 * 0.8
        assert result == pytest.approx(expected)

    def test_custom_weight_scheme_060202(self):
        """Phi sensitivity: weights (0.6, 0.2, 0.2)."""
        result = compute_phi(0.6, 0.7, 0.8, w_cov=0.6, w_step=0.2, w_tau=0.2)
        expected = 0.6 * 0.6 + 0.2 * 0.7 + 0.2 * 0.8
        assert result == pytest.approx(expected)

    def test_unrounded_preserves_precision(self):
        """Default (unrounded) retains full precision for repeating decimals."""
        result = compute_phi(1/3, 0.0, 0.0)
        assert result == pytest.approx(1/6)

    def test_round_result_true(self):
        """round_result=True rounds to 3 decimal places via round3()."""
        result = compute_phi(1/3, 0.0, 0.0, round_result=True)
        assert result == pytest.approx(0.167)

    def test_edge_case_all_zeros(self):
        """All zero inputs produce zero output."""
        result = compute_phi(0.0, 0.0, 0.0)
        assert result == 0.0

    def test_edge_case_perfect_scores(self):
        """All 1.0 produces Phi = 1.0 (regardless of weights)."""
        result = compute_phi(1.0, 1.0, 1.0)
        assert result == 1.0
        result2 = compute_phi(1.0, 1.0, 1.0, w_cov=0.4, w_step=0.4, w_tau=0.2)
        assert result2 == 1.0
