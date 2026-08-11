"""
Pins the degeneracy story of the synthetic clouds in data/.

demo_degeneracy.py and the README claim that the staircase constrains
exactly five DOFs (everything but cross-step translation ty) and the
flat plane exactly three (tz, roll, pitch). These tests hold the
generators — and the committed .pcd files — to that claim, with margin
assertions so a normals/threshold drift shows up here rather than as a
silently broken demo.
"""
import importlib.util
import os

import numpy as np
import pytest

from point_cloud_registration import PlaneICP, analyse_hessian

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data")

_spec = importlib.util.spec_from_file_location(
    "generate_synthetic", os.path.join(DATA_DIR, "generate_synthetic.py"))
generate_synthetic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generate_synthetic)


def _analyse(points):
    """Degeneracy analysis of a cloud registered against itself at identity."""
    engine = PlaneICP(max_iter=1, max_dist=1.0, k=10)
    engine.set_target(points)
    H = engine.calc_H_g_e2(np.eye(4), points.astype(np.float32))[0]
    return analyse_hessian(H.astype(float))


def _dominant_dof(eigenvector):
    return int(np.argmax(np.abs(eigenvector)))


def test_staircase_flags_exactly_ty():
    deg = _analyse(generate_synthetic.make_staircase())

    assert deg.num_constrained_dof == 5
    flagged = np.where(deg.degenerate_mask)[0]
    assert flagged.shape == (1,)
    # The single degenerate direction is dominated by ty (DOF index 1).
    assert _dominant_dof(deg.eigenvectors[:, flagged[0]]) == 1
    # Margins: the flagged eigenvalue sits clearly below the threshold and
    # the next one clearly above, so small sampling/normals drift cannot
    # silently flip the classification.
    assert deg.eigenvalues[flagged[0]] < 0.8 * deg.lambda_threshold
    assert deg.eigenvalues[1] > 5.0 * deg.lambda_threshold


def test_plane_flags_three_dof():
    deg = _analyse(generate_synthetic.make_plane())

    assert deg.num_constrained_dof == 3
    flagged = np.where(deg.degenerate_mask)[0]
    doms = sorted(_dominant_dof(deg.eigenvectors[:, i]) for i in flagged)
    assert doms == [0, 1, 5]  # tx, ty, yaw
    assert np.all(deg.eigenvalues[flagged] < 0.8 * deg.lambda_threshold)


def test_generators_are_deterministic():
    a = generate_synthetic.make_staircase()
    b = generate_synthetic.make_staircase()
    np.testing.assert_array_equal(a, b)

    c = generate_synthetic.make_plane()
    d = generate_synthetic.make_plane()
    np.testing.assert_array_equal(c, d)


def test_pcd_roundtrip(tmp_path):
    points = generate_synthetic.make_plane(num_points=100)
    path = tmp_path / "roundtrip.pcd"
    generate_synthetic.save_pcd(path, points)
    loaded = generate_synthetic.load_pcd(path)
    # save_pcd documents float32 / six-decimal storage: measured worst-case
    # roundtrip error across all shipped clouds is ~7.3e-7.
    np.testing.assert_allclose(loaded, points, atol=2e-6)


@pytest.mark.parametrize("name,expected_degenerate_doms", [
    ("synthetic_staircase_target.pcd", [1]),
    ("synthetic_staircase_source.pcd", [1]),
    ("synthetic_plane_target.pcd", [0, 1, 5]),
    ("synthetic_plane_source.pcd", [0, 1, 5]),
])
def test_committed_pcds_keep_their_degeneracy(name, expected_degenerate_doms):
    """
    The files on disk — not just fresh arrays — carry the story: the same
    DOF identities and the same classification margin, for every committed
    cloud (each has its own seed and therefore its own margin).
    """
    path = os.path.join(DATA_DIR, name)
    points = generate_synthetic.load_pcd(path)
    deg = _analyse(points)

    assert deg.num_constrained_dof == 6 - len(expected_degenerate_doms)
    flagged = np.where(deg.degenerate_mask)[0]
    doms = sorted(_dominant_dof(deg.eigenvectors[:, i]) for i in flagged)
    assert doms == expected_degenerate_doms
    assert np.all(deg.eigenvalues[flagged] < 0.8 * deg.lambda_threshold), (
        f"weak margin: {deg.eigenvalues[flagged]} vs threshold "
        f"{deg.lambda_threshold}"
    )
