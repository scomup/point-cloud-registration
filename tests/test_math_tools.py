"""
expSO3 must return a member of SO(3) in both of its branches.

The small-angle branch used to return the first-order I + W, which is
not orthonormal (det = 1 + O(theta^2)): at |omega| = 3e-3 the
determinant error is ~9e-6, and every Gauss-Newton step taken through
plus() bakes that scale/shear into the pose estimate.
"""
import numpy as np

from point_cloud_registration.math_tools import expSO3, skew


def _rodrigues(omega):
    """Exact Rodrigues formula, valid for any nonzero angle (float64)."""
    theta = np.linalg.norm(omega)
    K = skew(omega) / theta
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _assert_in_SO3(R, atol):
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=atol)
    assert abs(np.linalg.det(R) - 1.0) < atol


def test_small_angle_branch_is_orthonormal():
    # theta^2 = 9e-6 <= epsilon = 1e-5: exercises the near-zero branch.
    R = expSO3(np.array([3e-3, 0.0, 0.0]))
    _assert_in_SO3(R, atol=1e-12)


def test_small_angle_branch_matches_rodrigues():
    omega = np.array([1.5e-3, -2e-3, 1e-3])
    np.testing.assert_allclose(expSO3(omega), _rodrigues(omega), atol=1e-12)


def test_large_angle_branch_is_orthonormal():
    R = expSO3(np.array([0.5, -0.3, 0.2]))
    _assert_in_SO3(R, atol=1e-12)


def test_zero_rotation_is_identity():
    np.testing.assert_allclose(expSO3(np.zeros(3)), np.eye(3), atol=1e-15)
