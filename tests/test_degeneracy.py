"""
Unit tests for point_cloud_registration.degeneracy.

Hessian fixtures:
  - 1-DOF: diag([ε,ε,λ_big,ε,ε,ε]) — only tz constrained
  - 2-DOF: diag([λ_med,ε,λ_big,ε,ε,ε]) — tx + tz constrained
  - well-conditioned: A.T@A + 100*I — all 6 DOFs constrained
"""
import numpy as np
import pytest

from point_cloud_registration.degeneracy import (
    DegeneracyResult,
    analyse_hessian,
    apply_sr_solve,
)

# PCR DOF order: [tx, ty, tz, ωx, ωy, ωz]
#
# For a direction to be non-degenerate under the condition-number criterion:
#   lambda_i >= lambda_cn = sqrt(lambda_max / lambda_min)
# With lambda_min = EPS = 1e-6 and lambda_max = LAMBDA_BIG = 1e8:
#   lambda_cn = sqrt(1e8/1e-6) = 1e7
# So LAMBDA_BIG = 1e8 >= 1e7 → non-degenerate; EPS = 1e-6 << 1e7 → degenerate.
# LAMBDA_MED = 5e7 >= 1e7 → also non-degenerate (used for 2-DOF case).
EPS = 1e-6
LAMBDA_BIG = 1e8
LAMBDA_MED = 5e7


def _h_1dof() -> np.ndarray:
    """Only tz constrained — index 2 in PCR order."""
    return np.diag([EPS, EPS, LAMBDA_BIG, EPS, EPS, EPS])


def _h_2dof() -> np.ndarray:
    """tx (index 0) and tz (index 2) constrained."""
    return np.diag([LAMBDA_MED, EPS, LAMBDA_BIG, EPS, EPS, EPS])


def _h_well_conditioned() -> np.ndarray:
    rng = np.random.default_rng(42)
    A = rng.standard_normal((6, 6))
    return A.T @ A + 100.0 * np.eye(6)


class TestAnalyseHessian:
    def test_1dof_constrained_count(self):
        deg = analyse_hessian(_h_1dof())
        assert deg.num_constrained_dof == 1

    def test_2dof_constrained_count(self):
        deg = analyse_hessian(_h_2dof())
        assert deg.num_constrained_dof == 2

    def test_well_conditioned_no_degeneracy(self):
        deg = analyse_hessian(_h_well_conditioned())
        assert not deg.is_degenerate
        assert deg.num_constrained_dof == 6

    def test_1dof_is_degenerate(self):
        assert analyse_hessian(_h_1dof()).is_degenerate

    def test_condition_number_formula(self):
        # lambda_max = 100, lambda_min = 1 → cn = sqrt(100/1) = 10
        H = np.diag([1.0, 1.0, 100.0, 1.0, 1.0, 1.0])
        deg = analyse_hessian(H)
        expected = np.sqrt(100.0 / 1.0)
        assert abs(deg.condition_number - expected) < 0.01

    def test_eigenvalues_ascending(self):
        deg = analyse_hessian(_h_1dof())
        assert np.all(np.diff(deg.eigenvalues) >= 0)

    def test_V_constrained_shape(self):
        deg = analyse_hessian(_h_1dof())
        assert deg.V_constrained.shape == (1, 6)

        deg2 = analyse_hessian(_h_2dof())
        assert deg2.V_constrained.shape == (2, 6)

        deg6 = analyse_hessian(_h_well_conditioned())
        assert deg6.V_constrained.shape == (6, 6)

    def test_V_constrained_rows_are_unit_vectors(self):
        deg = analyse_hessian(_h_2dof())
        norms = np.linalg.norm(deg.V_constrained, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_fixed_lambda_threshold(self):
        H = _h_1dof()
        deg_low = analyse_hessian(H, lambda_threshold=EPS / 2)
        # Threshold below all eigenvalues → nothing degenerate
        assert deg_low.num_constrained_dof == 6

    def test_completely_degenerate(self):
        H = np.zeros((6, 6))
        deg = analyse_hessian(H)
        assert deg.is_degenerate
        assert deg.num_constrained_dof == 0
        assert deg.V_constrained.shape == (0, 6)

    def test_structural_zero_alone_is_degenerate(self):
        """
        A structural zero stays degenerate even when the adaptive threshold
        lands below it: five equal 1e12 eigenvalues give a threshold of
        sqrt(1e12/1e12) = 1, yet the 10.0 direction is eleven orders of
        magnitude weaker and must not be reported constrained.
        """
        H = np.diag([1e12, 1e12, 1e12, 1e12, 1e12, 10.0])
        deg = analyse_hessian(H)
        assert deg.num_constrained_dof == 5
        assert deg.degenerate_mask[0]  # ascending order puts 10.0 first
        assert deg.condition_number == pytest.approx(1.0)

    def test_structural_zero_combines_with_cn_test(self):
        """Phase 1 (structural zeros) and phase 2 (cn test) OR together."""
        H = np.diag([1e12, 1e12, 1e12, 1e12, 1e3, 1e-6])
        deg = analyse_hessian(H)
        # 1e-6 is a structural zero; 1e3 fails the cn test (~3.2e4).
        assert deg.num_constrained_dof == 4


class TestApplySRSolve:
    def test_degenerate_direction_zeroed(self):
        H = _h_1dof()
        deg = analyse_hessian(H)
        g = np.ones(6)
        dx = apply_sr_solve(H, g, deg)

        # Component of dx along each degenerate eigenvector should be ~0
        for i in range(6):
            if deg.degenerate_mask[i]:
                v = deg.eigenvectors[:, i]
                component = float(v @ dx)
                assert abs(component) < 1e-8, f"Degenerate dir {i} not zeroed: {component}"

    def test_no_degeneracy_matches_standard_solve(self):
        H = _h_well_conditioned()
        deg = analyse_hessian(H)
        g = np.ones(6)
        dx_sr = apply_sr_solve(H, g, deg)
        dx_std = np.linalg.solve(H, -g)
        np.testing.assert_allclose(dx_sr, dx_std, rtol=1e-6)

    def test_exactly_singular_hessian_classified_but_solve_raises(self):
        """
        Pins the contract boundary: analyse_hessian CLASSIFIES an exactly
        rank-deficient Hessian, but apply_sr_solve does not solve one.

        H is singular yet partially constrained (only tz observed).  The
        analysis correctly reports 1 constrained DOF, but the remapping still
        goes through np.linalg.solve(H, g), which raises before any projection
        happens.  Callers whose raw Hessian can be exactly singular must pass a
        regularized H to apply_sr_solve while analysing the raw one.
        """
        H = np.diag([0.0, 0.0, LAMBDA_BIG, 0.0, 0.0, 0.0])
        deg = analyse_hessian(H)
        assert deg.num_constrained_dof == 1

        with pytest.raises(np.linalg.LinAlgError):
            apply_sr_solve(H, np.ones(6), deg)
