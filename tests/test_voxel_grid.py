"""
Tests for VoxelGrid covariance regularization (cov_reg), the explicit
raise when the min_points mask empties the voxel set, and the threading of
both knobs through VPlaneICP and NDT.
"""

import numpy as np
import pytest

from point_cloud_registration.voxel import VoxelGrid
from point_cloud_registration.voxelized_plane_icp import VPlaneICP
from point_cloud_registration.ndt import NDT

VOXEL_SIZE = 1.0
COV_REG = 1e-3


@pytest.fixture
def coplanar_cloud():
    """
    A dense flat z=0 grid, spacing 0.2 over [0, 4).  With voxel_size 1.0 every
    voxel holds 25 exactly coplanar points, so each per-voxel covariance is
    exactly singular (smallest eigenvalue 0) and NDT's fast inverse hits its
    det_A == 0 clamp.
    """
    g = np.arange(0.0, 4.0, 0.2)
    xx, yy = np.meshgrid(g, g)
    return np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])


@pytest.fixture
def dense_cloud():
    """A well-conditioned random cloud: 4000 points filling 64 voxels."""
    rng = np.random.default_rng(7)
    return rng.uniform(0.0, 4.0, size=(4000, 3))


@pytest.fixture
def sparse_cloud():
    """
    Six voxels holding four points each.  min_points=10 masks every voxel away
    (empty voxel set); min_points=3 keeps all six.
    """
    return np.array(
        [[i * 2.0 + 0.1 * j, i * 2.0, 0.3 * j] for i in range(6) for j in range(4)],
        dtype=float,
    )


@pytest.fixture
def tiny_two_voxel_cloud():
    """Eight points in two voxels — small enough to check covariances by hand."""
    return np.array([
        [0.1, 0.1, 0.1], [0.2, 0.4, 0.3], [0.5, 0.2, 0.7], [0.3, 0.9, 0.2],
        [1.1, 0.1, 0.1], [1.4, 0.6, 0.3], [1.9, 0.2, 0.8], [1.3, 0.7, 0.4],
    ], dtype=float)


@pytest.fixture
def bumpy_pair():
    """
    Three gently bumpy, mutually orthogonal faces ("corner"), plus a purely
    translated copy.  Normals span R^3, so all six DOF are constrained.

    Returns (target, source, T_true) where T_true is the transform align()
    should recover, i.e. the one mapping source back onto target.
    """
    rng = np.random.default_rng(3)
    n = 3000
    a = rng.uniform(0.0, 4.0, size=(n, 2))
    b = rng.uniform(0.0, 4.0, size=(n, 2))
    c = rng.uniform(0.0, 4.0, size=(n, 2))

    def bump(u, v):
        return 0.05 * np.sin(u) * np.cos(v)

    face_z = np.column_stack([a[:, 0], a[:, 1], bump(a[:, 0], a[:, 1])])
    face_x = np.column_stack([bump(b[:, 0], b[:, 1]), b[:, 0], b[:, 1]])
    face_y = np.column_stack([c[:, 0], bump(c[:, 0], c[:, 1]), c[:, 1]])
    target = np.vstack([face_z, face_x, face_y])

    t = np.array([0.12, -0.09, 0.07])
    source = target + t
    T_true = np.eye(4)
    T_true[:3, 3] = -t
    return target, source, T_true


# --------------------------------------------------------------------------
# 1. cov_reg regularizes coplanar voxels
# --------------------------------------------------------------------------

def test_cov_reg_regularizes_coplanar_voxels(coplanar_cloud):
    """
    An isotropic +cov_reg*I shift lifts the null direction of an exactly
    coplanar voxel, so the fast inverse becomes a true inverse instead of
    hitting the det_A == 0 clamp.
    """
    # Fixture guard: without regularization the covariances really are singular.
    plain = VoxelGrid(VOXEL_SIZE, min_points=3)
    plain.set_points(coplanar_cloud)
    assert np.allclose(np.linalg.eigvalsh(plain.cov)[:, 0], 0.0, atol=1e-12), (
        "fixture is not coplanar; smallest eigenvalues "
        f"{np.linalg.eigvalsh(plain.cov)[:, 0]}"
    )

    vg = VoxelGrid(VOXEL_SIZE, min_points=3, cov_reg=COV_REG)
    vg.set_points(coplanar_cloud)
    vg.calc_icov()

    assert np.all(np.isfinite(vg.icov)), "icov has non-finite entries"

    eigenvalues = np.linalg.eigvalsh(vg.cov)
    assert np.all(eigenvalues >= COV_REG - 1e-12), (
        f"covariance eigenvalues fell below the shift: min {eigenvalues.min()}"
    )

    # The clamp was not taken: icov really inverts cov.
    identity = np.broadcast_to(np.eye(3), vg.cov.shape)
    assert np.allclose(vg.icov @ vg.cov, identity, atol=1e-6), (
        "icov is not the inverse of cov; the singular clamp was still hit"
    )


def test_cov_reg_preserves_normals(coplanar_cloud):
    """An isotropic shift leaves the eigenvectors — and so the normals — intact."""
    plain = VoxelGrid(VOXEL_SIZE, min_points=3)
    plain.set_points(coplanar_cloud)

    regularized = VoxelGrid(VOXEL_SIZE, min_points=3, cov_reg=COV_REG)
    regularized.set_points(coplanar_cloud)

    assert np.allclose(np.abs(plain.norm), np.abs(regularized.norm), atol=1e-12), (
        "cov_reg rotated the voxel normals"
    )


# --------------------------------------------------------------------------
# 2. cov_reg=0 preserves upstream values exactly
# --------------------------------------------------------------------------

def test_default_cov_reg_equals_explicit_zero(dense_cloud):
    """The default must be bit-for-bit the unregularized path."""
    default = VoxelGrid(VOXEL_SIZE, min_points=3)
    default.set_points(dense_cloud)

    explicit = VoxelGrid(VOXEL_SIZE, min_points=3, cov_reg=0.0)
    explicit.set_points(dense_cloud)

    assert default.cov_reg == 0.0
    assert np.array_equal(default.cov, explicit.cov)
    assert np.array_equal(default.mean, explicit.mean)
    assert np.array_equal(default.norm, explicit.norm)


def test_negative_cov_reg_is_rejected():
    """A negative shift would make covariances indefinite: reject, not ignore."""
    with pytest.raises(ValueError, match="cov_reg"):
        VoxelGrid(VOXEL_SIZE, cov_reg=-1e-3)


def test_default_cov_reg_matches_hand_computed_statistics(tiny_two_voxel_cloud):
    """
    With cov_reg at its default the per-voxel mean/covariance/normal must equal
    the textbook values (ddof=1 sample covariance), i.e. upstream behavior.
    """
    vg = VoxelGrid(VOXEL_SIZE, min_points=3)
    vg.set_points(tiny_two_voxel_cloud)

    assert len(vg.mean) == 2

    for i, sl in enumerate([slice(0, 4), slice(4, 8)]):
        points = tiny_two_voxel_cloud[sl]
        expected_mean = points.mean(axis=0)
        expected_cov = np.cov(points.T, ddof=1)
        expected_norm = np.linalg.eigh(expected_cov)[1][:, 0]

        assert np.allclose(vg.mean[i], expected_mean, atol=1e-12)
        assert np.allclose(vg.cov[i], expected_cov, atol=1e-12)
        assert np.allclose(np.abs(vg.norm[i]), np.abs(expected_norm), atol=1e-9)


# --------------------------------------------------------------------------
# 3. raise-on-empty
# --------------------------------------------------------------------------

def test_set_points_raises_when_min_points_empties_the_grid(sparse_cloud):
    """
    Every voxel holds four points, so min_points=10 leaves nothing behind.  The
    empty grid must be reported explicitly rather than relying on whichever
    KDTree backend happens to be compiled in.
    """
    with pytest.raises(ValueError, match="data_pts should be non-empty"):
        VoxelGrid(VOXEL_SIZE, min_points=10).set_points(sparse_cloud)


def test_set_points_succeeds_when_min_points_admits_voxels(sparse_cloud):
    """The same cloud with a reachable threshold builds a usable grid."""
    vg = VoxelGrid(VOXEL_SIZE, min_points=3)
    vg.set_points(sparse_cloud)

    assert len(vg.mean) > 0
    assert len(vg.mean) == len(vg.cov) == len(vg.norm)


# --------------------------------------------------------------------------
# 4. VPlaneICP threading
# --------------------------------------------------------------------------

def test_vplane_icp_threads_min_points_and_cov_reg(sparse_cloud):
    """Both knobs must reach the VoxelGrid that set_target builds."""
    engine = VPlaneICP(voxel_size=VOXEL_SIZE, min_points=3, cov_reg=COV_REG)
    engine.set_target(sparse_cloud)

    assert engine.is_target_set()
    assert len(engine.voxels.mean) > 0
    assert engine.voxels.min_points == 3
    assert engine.voxels.cov_reg == COV_REG


def test_vplane_icp_propagates_empty_voxel_error(sparse_cloud):
    """An unreachable min_points surfaces as a ValueError through set_target."""
    engine = VPlaneICP(voxel_size=VOXEL_SIZE, min_points=10)
    with pytest.raises(ValueError, match="data_pts should be non-empty"):
        engine.set_target(sparse_cloud)


def test_vplane_icp_defaults_preserve_upstream(dense_cloud):
    """Untouched constructor arguments keep the upstream (10, 0.0) behavior."""
    engine = VPlaneICP(voxel_size=VOXEL_SIZE)
    engine.set_target(dense_cloud)

    assert engine.voxels.min_points == 10
    assert engine.voxels.cov_reg == 0.0


# --------------------------------------------------------------------------
# 5. NDT threading
# --------------------------------------------------------------------------

def test_ndt_threads_cov_reg_for_coplanar_target(coplanar_cloud):
    """
    NDT.set_target runs calc_icov itself, so cov_reg has to be in place by then;
    with it the inverse covariances are finite, moderate and genuine inverses.
    """
    engine = NDT(voxel_size=VOXEL_SIZE, min_points=3, cov_reg=COV_REG)
    engine.set_target(coplanar_cloud)

    assert engine.is_target_set()
    assert engine.voxels.min_points == 3
    assert engine.voxels.cov_reg == COV_REG
    assert np.all(np.isfinite(engine.voxels.icov))

    identity = np.broadcast_to(np.eye(3), engine.voxels.cov.shape)
    assert np.allclose(engine.voxels.icov @ engine.voxels.cov, identity, atol=1e-6)

    # The null direction is now weighted at ~1/cov_reg rather than clamped away.
    max_icov = np.max(np.abs(engine.voxels.icov))
    assert 1e2 < max_icov < 1e4, f"icov magnitude out of range: {max_icov}"


def test_ndt_defaults_hit_singular_clamp_on_coplanar_target(coplanar_cloud):
    """
    Without cov_reg the det_A == 0 clamp replaces the determinant with 1e6, which
    collapses icov to ~0: the voxel contributes no information at all.  This is
    the behavior cov_reg exists to avoid, and it must remain the default.
    """
    engine = NDT(voxel_size=VOXEL_SIZE)
    engine.set_target(coplanar_cloud)

    assert engine.voxels.cov_reg == 0.0

    icov = engine.voxels.icov
    identity = np.broadcast_to(np.eye(3), engine.voxels.cov.shape)
    assert not np.allclose(icov @ engine.voxels.cov, identity, atol=1e-6), (
        "clamped icov unexpectedly inverts cov"
    )
    assert np.max(np.abs(icov)) < 1e-6, (
        f"expected the clamp to collapse icov, got max {np.max(np.abs(icov))}"
    )

    # Contrast: the regularized grid carries ~9 orders of magnitude more weight.
    regularized = NDT(voxel_size=VOXEL_SIZE, min_points=3, cov_reg=COV_REG)
    regularized.set_target(coplanar_cloud)
    assert np.max(np.abs(regularized.voxels.icov)) > 1e6 * np.max(np.abs(icov))


# --------------------------------------------------------------------------
# 6. end-to-end sanity
# --------------------------------------------------------------------------

def test_vplane_icp_end_to_end_alignment_with_cov_reg(bumpy_pair):
    """Threading the new knobs must not disturb registration itself."""
    target, source, T_true = bumpy_pair

    engine = VPlaneICP(voxel_size=0.5, max_iter=50, max_dist=2.0, tol=1e-6,
                       min_points=3, cov_reg=COV_REG)
    engine.set_target(target)

    T = engine.align(source)

    assert np.all(np.isfinite(T))
    assert np.allclose(T, T_true, atol=2e-2), (
        f"max diff from ground truth {np.max(np.abs(T - T_true))}"
    )
