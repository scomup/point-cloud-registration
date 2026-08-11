"""
Behavioral tests for Registration.align().
"""
import numpy as np
import pytest

from point_cloud_registration.icp import ICP
from point_cloud_registration.plane_icp import PlaneICP
from point_cloud_registration.math_tools import expSO3, makeT


@pytest.fixture
def well_conditioned_pair():
    """
    Three mutually orthogonal planes ("corner") — normals span R^3, so all six
    DOF are constrained and the Hessian is well conditioned.

    Returns (target, source, T_true) where T_true is the transform align()
    should recover, i.e. the one that maps source onto target.
    """
    rng = np.random.default_rng(0)
    n = 400
    a = rng.uniform(0.0, 4.0, size=(n, 2))
    b = rng.uniform(0.0, 4.0, size=(n, 2))
    c = rng.uniform(0.0, 4.0, size=(n, 2))
    face_z = np.column_stack([a[:, 0], a[:, 1], np.zeros(n)])
    face_x = np.column_stack([np.zeros(n), b[:, 0], b[:, 1]])
    face_y = np.column_stack([c[:, 0], np.zeros(n), c[:, 1]])
    target = np.vstack([face_z, face_x, face_y])

    R = expSO3(np.array([0.02, -0.03, 0.05]))
    t = np.array([0.15, -0.1, 0.08])
    source = (R @ target.T).T + t
    T_true = np.linalg.inv(makeT(R, t))
    return target, source, T_true


@pytest.fixture
def degenerate_pair():
    """
    A perfectly flat z=0 grid as target, source the same grid shifted in +z.

    Every surface normal is (0, 0, 1), so the plane-ICP Jacobian rows are
    [0, 0, 1, y, -x, 0]: tx, ty and wz never appear and the Hessian is exactly
    rank 3.  z, wx and wy are the only observable directions.
    """
    g = np.arange(-5.0, 5.0 + 1e-9, 1.0)
    xx, yy = np.meshgrid(g, g)
    target = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    source = target + np.array([0.0, 0.0, 0.5])
    return target, source


def _yaw(T):
    return float(np.arctan2(T[1, 0], T[0, 0]))


def test_last_hessian_lifecycle(well_conditioned_pair):
    """last_hessian is None before any align and a 6x6 array afterwards."""
    target, source, _ = well_conditioned_pair
    engine = PlaneICP(max_iter=30, max_dist=2.0, tol=1e-6)
    engine.set_target(target)

    assert engine.last_hessian is None

    engine.align(source)

    H = engine.last_hessian
    assert isinstance(H, np.ndarray)
    assert H.shape == (6, 6)
    assert np.all(np.isfinite(H))


def test_last_hessian_recomputed_when_max_iter_exhausted(well_conditioned_pair):
    """
    When align() runs out of iterations, cur_T has advanced past the last
    linearization, so the stored Hessian must be recomputed at the returned
    pose rather than left at the pre-step one.
    """
    target, source, _ = well_conditioned_pair
    engine = PlaneICP(max_iter=1, max_dist=2.0, tol=1e-6)
    engine.set_target(target)

    T = engine.align(source)

    H_at_returned = engine.calc_H_g_e2(T, source)[0]
    H_at_init = engine.calc_H_g_e2(np.eye(4), source)[0]

    assert np.allclose(engine.last_hessian, H_at_returned), (
        "last_hessian is not the Hessian at the returned pose; max diff "
        f"{np.max(np.abs(engine.last_hessian - H_at_returned))}"
    )
    assert not np.allclose(engine.last_hessian, H_at_init), (
        "last_hessian is stale (equals the pre-step Hessian)"
    )


def test_align_does_not_alias_init_T():
    """
    align() must return a transform the caller owns.

    With source == target the very first step is ~zero, so align()
    converges before ever calling plus(): without the defensive copy it
    returns the init_T object itself — and with the mutable np.eye(4)
    default, a caller mutating the result silently corrupts the default
    for every subsequent align() call in the process.
    """
    np.random.seed(1)
    target = np.random.rand(100, 3)
    icp = ICP(max_iter=10, max_dist=2.0, tol=1e-3)
    icp.set_target(target)
    source = target.astype(np.float32)

    init_T = np.eye(4)
    T = icp.align(source, init_T=init_T)
    assert T is not init_T

    # Mutating the result must not corrupt the shared default argument.
    T_default = icp.align(source)
    T_default[0, 3] = 123.0
    T_again = icp.align(source)
    np.testing.assert_allclose(T_again, np.eye(4), atol=1e-6)


def test_lm_damping_rescues_singular_hessian(degenerate_pair):
    """
    On exactly rank-deficient geometry the plain Gauss-Newton solve fails; the
    damped solve returns a finite transform.
    """
    target, source = degenerate_pair

    engine = PlaneICP(max_iter=10, max_dist=2.0, tol=1e-6)
    engine.set_target(target)

    # The fixture really is singular: three eigenvalues are exactly zero.
    H = engine.calc_H_g_e2(np.eye(4), source)[0]
    eigenvalues = np.linalg.eigvalsh(H)
    assert np.count_nonzero(eigenvalues <= 0.0) == 3, f"eigenvalues: {eigenvalues}"

    # Undamped: np.linalg.solve rejects the exactly singular Hessian.
    with pytest.raises(np.linalg.LinAlgError):
        engine.align(source, lm_damping=False)

    # Damped: H + lambda*I is invertible, so align completes.
    T = engine.align(source, lm_damping=True)
    assert T.shape == (4, 4)
    assert np.all(np.isfinite(T))


def test_lm_damping_parity_on_well_conditioned_data(well_conditioned_pair):
    """Damping must not move the solution when the problem is well conditioned."""
    target, source, _ = well_conditioned_pair

    engine = PlaneICP(max_iter=30, max_dist=2.0, tol=1e-6)
    engine.set_target(target)
    T_plain = engine.align(source, lm_damping=False)

    engine_damped = PlaneICP(max_iter=30, max_dist=2.0, tol=1e-6)
    engine_damped.set_target(target)
    T_damped = engine_damped.align(source, lm_damping=True)

    assert np.allclose(T_plain, T_damped, atol=1e-3), (
        f"damping changed the solution; max diff {np.max(np.abs(T_plain - T_damped))}"
    )


def test_solution_remapping_zeroes_degenerate_directions(degenerate_pair):
    """
    On a flat plane the observable directions are z, wx and wy.  Solution
    remapping must leave the unobservable ones (x, y, yaw) untouched while z
    still converges onto the target plane.

    lm_damping is required here: apply_sr_solve inverts the Hessian it is given,
    and the raw one is singular.
    """
    target, source = degenerate_pair

    engine = PlaneICP(max_iter=30, max_dist=2.0, tol=1e-6)
    engine.set_target(target)

    T = engine.align(source, use_solution_remapping=True, lm_damping=True)

    assert np.all(np.isfinite(T))
    assert abs(T[0, 3]) < 1e-6, f"x moved along a degenerate direction: {T[0, 3]}"
    assert abs(T[1, 3]) < 1e-6, f"y moved along a degenerate direction: {T[1, 3]}"
    assert abs(_yaw(T)) < 1e-6, f"yaw moved along a degenerate direction: {_yaw(T)}"
    # The source sits 0.5 above the target plane, so align must pull it back down.
    assert T[2, 3] == pytest.approx(-0.5, abs=1e-3), f"z did not converge: {T[2, 3]}"


def test_solution_remapping_no_world_leak_under_rotated_init(degenerate_pair):
    """
    The same flat plane, but starting from a rotated initial guess.

    The unobservable directions of the scene are world x, y and yaw; SR zeroes
    the step in the body frame, and because H and the step share the body
    frame of the retraction, the accumulated world-frame motion must still
    have no component along them.  Before the body-frame Jacobian fix this
    leaked ~0.13 m into world y at a 0.25 rad initial roll.
    """
    from point_cloud_registration.math_tools import transform_points

    target, source = degenerate_pair

    engine = PlaneICP(max_iter=50, max_dist=2.0, tol=1e-8)
    engine.set_target(target)

    init_T = makeT(expSO3(np.array([0.25, 0.0, 0.0])), np.zeros(3))
    T = engine.align(source, init_T=init_T,
                     use_solution_remapping=True, lm_damping=True)

    assert np.all(np.isfinite(T))
    assert abs(T[0, 3]) < 1e-6, f"x leaked: {T[0, 3]}"
    assert abs(T[1, 3]) < 1e-6, f"y leaked: {T[1, 3]}"
    assert abs(_yaw(T)) < 1e-6, f"yaw leaked: {_yaw(T)}"
    # The observable directions must still do their job: the aligned source
    # has to land on the z=0 target plane.
    aligned = transform_points(T, source)
    assert np.max(np.abs(aligned[:, 2])) < 1e-3


def test_default_behaviour_converges_to_ground_truth(well_conditioned_pair):
    """With both options off, align() is the plain Gauss-Newton solver."""
    target, source, T_true = well_conditioned_pair

    engine = PlaneICP(max_iter=30, max_dist=2.0, tol=1e-6)
    engine.set_target(target)

    T = engine.align(source, use_solution_remapping=False, lm_damping=False)

    assert T.shape == (4, 4)
    assert np.allclose(T, T_true, atol=1e-4), (
        f"max diff from ground truth {np.max(np.abs(T - T_true))}"
    )
