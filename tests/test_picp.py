import numpy as np
import pytest
from point_cloud_registration.plane_icp import PlaneICP
from point_cloud_registration.math_tools import expSO3


@pytest.fixture
def generate_test_data():
    """
    Generate synthetic data for testing ICP.
    """
    np.random.seed(42)
    target = np.random.rand(100, 3)  # 100 random points in 3D
    R = expSO3(np.array([0.1, 0.2, 0.3]))  # Small rotation
    t = np.array([0.5, -0.3, 0.2])  # Small translation
    source = (R @ target.T).T + t  # Apply transformation to target
    return target, source


def test_calc_H_g_e2(generate_test_data):
    """
    Test that calc_H_g_e2 and calc_H_g_e2_no_parallel_ver produce the same results.
    """
    target, source = generate_test_data
    source = source.astype(np.float32)
    vpicp = PlaneICP(max_iter=10, max_dist=2.0, tol=1e-6)
    vpicp.set_target(target)

    cur_T = np.eye(4)  # Initial transformation (identity matrix)

    # Compute results using both methods
    H1, g1, e2_1 = vpicp.calc_H_g_e2(cur_T, source)
    H2, g2, e2_2 = vpicp.calc_H_g_e2_no_parallel_ver(cur_T, source)

    # Assert that the results are the same
    assert np.allclose(H1, H2, atol=1e-6), f"H matrices differ: {np.max(np.abs(H1 - H2))}"
    assert np.allclose(g1, g2, atol=1e-6), f"g vectors differ: {np.max(np.abs(g1 - g2))}"
    assert np.isclose(e2_1, e2_2, atol=1e-6), f"e2 values differ: {abs(e2_1 - e2_2)}"
