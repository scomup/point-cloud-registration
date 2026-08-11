import numpy as np
import pytest
from point_cloud_registration import VPlaneICP
from point_cloud_registration import expSO3
from point_cloud_registration.math_tools import makeT


@pytest.fixture
def generate_test_data():
    """
    Generate synthetic data for testing PlaneICP.
    """
    np.random.seed(42)
    target = np.random.rand(100, 3)  # 100 random points in 3D
    normals = np.random.rand(100, 3)  # Random normals for target points
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)  # Normalize normals
    R = expSO3(np.array([0.1, 0.2, 0.3]))  # Small rotation
    t = np.array([0.5, -0.3, 0.2])  # Small translation
    source = (R @ target.T).T + t  # Apply transformation to target
    return target, normals, source


@pytest.mark.parametrize(
    "cur_T",
    [np.eye(4), makeT(expSO3(np.array([0.3, -0.2, 0.4])), np.array([0.1, 0.2, -0.1]))],
    ids=["identity_pose", "rotated_pose"],
)
@pytest.mark.parametrize("with_outliers", [False, True], ids=["all_inliers", "with_outliers"])
def test_calc_H_g_e2(generate_test_data, cur_T, with_outliers):
    """
    Test that calc_H_g_e2 and calc_H_g_e2_no_parallel_ver produce the same results.
    """
    target, _, source = generate_test_data
    if with_outliers:
        # Points far outside the target's reach: rejected by the max_dist gate.
        source = np.vstack([source[:10] + 50.0, source])
    source = source.astype(np.float32)
    plane_icp = VPlaneICP(voxel_size=1.0, max_iter=10, max_dist=2.0, tol=1e-3)
    plane_icp.set_target(target)


    # Compute results using both methods
    H1, g1, e2_1 = plane_icp.calc_H_g_e2(cur_T, source)
    H2, g2, e2_2 = plane_icp.calc_H_g_e2_no_parallel_ver(cur_T, source)

    # Assert that the results are the same
    assert np.allclose(H1, H2, atol=1e-3), f"H matrices differ: {np.max(np.abs(H1 - H2))}"
    assert np.allclose(g1, g2, atol=1e-3), f"g vectors differ: {np.max(np.abs(g1 - g2))}"
    assert np.isclose(e2_1, e2_2, atol=1e-3), f"e2 values differ: {abs(e2_1 - e2_2)}"
