"""
Retraction-consistency oracle for every solver's calc_H_g_e2.

align() applies its Gauss-Newton step through the right-multiplicative
retraction plus(T, dx) = T @ [expSO3(dx[3:]) | dx[:3]], so the (H, g)
returned by calc_H_g_e2 must be the Gauss-Newton pair of the solver's
own cost with respect to THAT increment (a body-frame right-tangent
6-vector, DOF order [tx, ty, tz, wx, wy, wz]).

The oracle freezes the correspondences the solver selected at cur_T
(Gauss-Newton linearizes with correspondences held fixed), rebuilds the
residual model in float64, differentiates each residual through plus()
by central differences, and checks

    g == sum_i J_i.T @ W_i @ r_i        H == sum_i J_i.T @ W_i @ J_i

independently.  This catches frame errors that fast-vs-reference parity
tests cannot see, because those compare two implementations that could
share the same wrong convention.
"""
import numpy as np
import pytest

from point_cloud_registration.icp import ICP
from point_cloud_registration.plane_icp import PlaneICP
from point_cloud_registration.voxelized_plane_icp import VPlaneICP
from point_cloud_registration.ndt import NDT
from point_cloud_registration.math_tools import expSO3, makeT, plus, transform_points

H_STEP = 3e-3
CUR_T = makeT(expSO3(np.array([0.3, -0.2, 0.4])), np.array([0.1, 0.2, -0.1]))


def _numeric_J(residual_fn, dim):
    """Central-difference Jacobian of residual_fn: R^6 -> R^(N,dim)."""
    r0 = residual_fn(np.zeros(6))
    J = np.zeros((r0.shape[0], dim, 6))
    for k in range(6):
        dx = np.zeros(6)
        dx[k] = H_STEP
        rp = residual_fn(dx)
        rm = residual_fn(-dx)
        J[:, :, k] = (rp - rm) / (2.0 * H_STEP)
    return r0, J


def _assert_H_g_match(H, g, r0, J, W=None):
    """Compare solver (H, g) with the numeric Gauss-Newton pair."""
    if W is None:
        Wr = r0
        WJ = J
    else:
        Wr = np.einsum('nij,nj->ni', W, r0)
        WJ = np.einsum('nij,njk->nik', W, J)
    g_num = np.einsum('nik,ni->k', J, Wr)
    H_num = np.einsum('nik,nil->kl', J, WJ)
    g_scale = max(np.max(np.abs(g_num)), 1.0)
    H_scale = max(np.max(np.abs(H_num)), 1.0)
    assert np.allclose(g, g_num, atol=2e-3 * g_scale), (
        f"g mismatch:\nsolver {g}\nnumeric {g_num}")
    assert np.allclose(H, H_num, atol=2e-3 * H_scale), (
        f"H mismatch: max |dH| = {np.max(np.abs(H - H_num))}")


@pytest.fixture
def cloud_pair():
    np.random.seed(42)
    target = np.random.rand(500, 3) * 4.0
    R = expSO3(np.array([0.1, 0.2, 0.3]))
    t = np.array([0.5, -0.3, 0.2])
    source = ((R @ target.T).T + t).astype(np.float32)
    return target, source


def test_icp_matches_numeric_gauss_newton(cloud_pair):
    target, source = cloud_pair
    icp = ICP(max_iter=10, max_dist=2.0)
    icp.set_target(target)
    H, g, _ = icp.calc_H_g_e2(CUR_T, source)

    # Freeze the correspondences exactly as the solver selected them.
    src_trans = transform_points(CUR_T.astype(np.float32), source)
    dist, idx = icp.kdtree.query(src_trans)
    mask = dist < icp.max_dist
    p = source[mask].astype(np.float64)
    q = target[idx[mask]].astype(np.float64)

    def residual(dx):
        T = plus(CUR_T, dx)
        return transform_points(T, p) - q

    r0, J = _numeric_J(residual, 3)
    _assert_H_g_match(H, g, r0, J)


def test_plane_icp_matches_numeric_gauss_newton(cloud_pair):
    target, source = cloud_pair
    picp = PlaneICP(max_iter=10, max_dist=2.0, k=10)
    picp.set_target(target)
    H, g, _ = picp.calc_H_g_e2(CUR_T, source)

    src_trans = transform_points(CUR_T.astype(np.float32), source)
    dist, idx = picp.kdtree.query(src_trans)
    mask = dist < picp.max_dist
    p = source[mask].astype(np.float64)
    m = picp.target[idx[mask]].astype(np.float64)
    n = picp.normal[idx[mask]].astype(np.float64)

    def residual(dx):
        T = plus(CUR_T, dx)
        return np.einsum('ij,ij->i', n, transform_points(T, p) - m)[:, None]

    r0, J = _numeric_J(residual, 1)
    _assert_H_g_match(H, g, r0, J)


def test_vplane_icp_matches_numeric_gauss_newton(cloud_pair):
    target, source = cloud_pair
    vpicp = VPlaneICP(voxel_size=1.0, max_iter=10, max_dist=2.0)
    vpicp.set_target(target)
    H, g, _ = vpicp.calc_H_g_e2(CUR_T, source)

    src_trans = transform_points(CUR_T.astype(np.float32), source)
    query = vpicp.voxels.query(src_trans, ['mean', 'norm'])
    mask = query['dist'] < vpicp.max_dist
    p = source[mask].astype(np.float64)
    m = query['mean'][mask].astype(np.float64)
    n = query['norm'][mask].astype(np.float64)

    def residual(dx):
        T = plus(CUR_T, dx)
        return np.einsum('ij,ij->i', n, transform_points(T, p) - m)[:, None]

    r0, J = _numeric_J(residual, 1)
    _assert_H_g_match(H, g, r0, J)


def test_ndt_matches_numeric_gauss_newton(cloud_pair):
    target, source = cloud_pair
    ndt = NDT(voxel_size=1.0, max_iter=10, max_dist=2.0)
    ndt.set_target(target)
    H, g, _ = ndt.calc_H_g_e2(CUR_T, source)

    src_trans = transform_points(CUR_T.astype(np.float32), source)
    query = ndt.voxels.query(src_trans, ['icov', 'mean'])
    mask = query['dist'] < ndt.max_dist
    p = source[mask].astype(np.float64)
    m = query['mean'][mask].astype(np.float64)
    W = query['icov'][mask].astype(np.float64)

    def residual(dx):
        T = plus(CUR_T, dx)
        return transform_points(T, p) - m

    r0, J = _numeric_J(residual, 3)
    _assert_H_g_match(H, g, r0, J, W=W)


class TestAlignRecovery:
    """End-to-end: align() must reach the true pose, not just a stationary one."""

    def test_icp_recovers_large_rotation(self):
        np.random.seed(7)
        target = np.random.rand(200, 3) * 2.0
        T_true = makeT(expSO3(np.array([0.5, 0.0, 0.0])),
                       np.array([0.3, -0.2, 0.1]))
        source = transform_points(np.linalg.inv(T_true), target)
        icp = ICP(max_iter=100, max_dist=5.0, tol=1e-9)
        icp.set_target(target)
        T = icp.align(source.astype(np.float32))
        assert np.allclose(T, T_true, atol=1e-3), (
            f"pose error {np.max(np.abs(T - T_true))}")

    def test_plane_icp_recovers_from_rotated_init(self):
        rng = np.random.default_rng(0)
        pts = []
        for _ in range(3):
            u = rng.uniform(0.0, 4.0, (400, 2))
            pts.append(np.column_stack([u[:, 0], u[:, 1], np.zeros(400)]))
        corner = np.vstack([
            pts[0],
            pts[1][:, [0, 2, 1]],
            pts[2][:, [2, 0, 1]],
        ])
        T_true = makeT(expSO3(np.array([0.02, -0.03, 0.05])),
                       np.array([0.15, -0.1, 0.08]))
        source = transform_points(np.linalg.inv(T_true), corner)
        picp = PlaneICP(max_iter=60, max_dist=1.0, tol=1e-8, k=10)
        picp.set_target(corner)
        init_T = makeT(expSO3(np.array([0.1, -0.05, 0.08])), np.zeros(3))
        T = picp.align(source.astype(np.float32), init_T=init_T)
        assert np.allclose(T, T_true, atol=1e-3), (
            f"pose error {np.max(np.abs(T - T_true))}")

    @pytest.mark.parametrize("engine_cls", [VPlaneICP, NDT])
    def test_voxel_solvers_reduce_pose_error(self, engine_cls):
        # Three gently textured faces of a corner, offset by +0.5 so the
        # surfaces sit mid-voxel: faces lying exactly on voxel boundaries
        # give voxels that straddle two faces and blend their normals,
        # which biases the voxelized cost minimum away from the true pose.
        rng = np.random.default_rng(3)
        faces = []
        for axes in ((0, 1, 2), (0, 2, 1), (2, 0, 1)):
            u = rng.uniform(0.0, 4.0, (3000, 2))
            face = np.zeros((3000, 3))
            face[:, axes[0]] = u[:, 0]
            face[:, axes[1]] = u[:, 1]
            face[:, axes[2]] = 0.05 * np.sin(u[:, 0]) * np.cos(u[:, 1])
            faces.append(face)
        target = np.vstack(faces) + 0.5
        T_true = makeT(expSO3(np.array([0.03, -0.02, 0.04])),
                       np.array([0.2, -0.15, 0.1]))
        source = transform_points(np.linalg.inv(T_true), target)
        engine = engine_cls(voxel_size=1.0, max_iter=60, max_dist=2.0, tol=1e-8)
        engine.set_target(target)
        T = engine.align(source.astype(np.float32))
        err_before = np.linalg.norm(np.eye(4) - T_true)
        err_after = np.linalg.norm(T - T_true)
        assert err_after < err_before / 10.0, (
            f"before {err_before}, after {err_after}")
        assert np.allclose(T[:3, 3], T_true[:3, 3], atol=3e-2)
