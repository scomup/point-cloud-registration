#!/usr/bin/env python3
"""
Demo: degeneracy detection + solution remapping on under-constrained scenes.

Two synthetic scenes from data/ (regenerated in memory if the .pcd files
are absent):

* staircase — treads and risers constrain everything except cross-step
  translation (ty): exactly one degenerate DOF.
* plane — a featureless flat seafloor: tx, ty and yaw are unobservable,
  three degenerate DOFs.

For each scene the source scan (an independent sampling of the same
surface) is displaced by a known ground-truth transform and aligned back
with PlaneICP twice: once plain (lm_damping only) and once with
use_solution_remapping=True. The demo prints the DegeneracyResult and a
per-DOF error table, and renders a 2x2 figure per case:

    A  the scans before alignment (3D)
    B  top-down view after alignment, with a zoom showing the drift
    C  the Hessian eigenvalue spectrum vs the SR threshold
    D  per-DOF |error| vs ground truth, plain vs SR

Without solution remapping the optimizer slides along the unobservable
directions, driven by nothing but sampling noise; with it, those
directions hold at the initial guess while the observable ones converge.

Usage:
    python3 demo_degeneracy.py                     # interactive windows
    python3 demo_degeneracy.py --save              # write imgs/degeneracy_*.png
    python3 demo_degeneracy.py --case stair --save

matplotlib is imported by this demo alone (pip install matplotlib); the
degeneracy feature adds no dependencies to the library.
"""
import argparse
import os
import sys

import numpy as np

from point_cloud_registration import PlaneICP, analyse_hessian
from point_cloud_registration.math_tools import expSO3, makeT, transform_points

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(REPO_DIR, "data")

DOF_LABELS = ["tx", "ty", "tz", "roll", "pitch", "yaw"]

# Okabe-Ito colorblind-safe colors (validated for CVD separation).
COLORS = {
    "target": "#999999",    # context layer
    "initial": "#E69F00",
    "baseline": "#D55E00",  # no solution remapping
    "sr": "#0072B2",        # with solution remapping
}

CASES = {
    "stair": {
        "title": "Staircase — 1 degenerate DOF (ty, cross-step)",
        "target_pcd": "synthetic_staircase_target.pcd",
        "source_pcd": "synthetic_staircase_source.pcd",
        "generator": ("make_staircase",
                      dict(seed=42, pts_per_face=220),
                      dict(seed=2024, pts_per_face=220)),
        # Ground truth deliberately has zero component along the degenerate
        # ty, so the SR-held value is also the correct one.
        "rvec": np.array([0.02, -0.015, 0.08]),
        "tvec": np.array([0.30, 0.0, -0.20]),
        "png": "degeneracy_stair.png",
    },
    "plane": {
        "title": "Flat plane — 3 degenerate DOFs (tx, ty, yaw)",
        "target_pcd": "synthetic_plane_target.pcd",
        "source_pcd": "synthetic_plane_source.pcd",
        "generator": ("make_plane",
                      dict(seed=7, num_points=2500),
                      dict(seed=99, num_points=2000)),
        "rvec": np.array([0.03, -0.02, 0.0]),
        "tvec": np.array([0.0, 0.0, 0.4]),
        "png": "degeneracy_plane.png",
    },
}


def _generator_module(data_dir):
    sys.path.insert(0, data_dir)
    try:
        import generate_synthetic
    finally:
        sys.path.pop(0)
    return generate_synthetic


def load_case(name, data_dir):
    """Return (target, source_surface), from disk or regenerated in memory."""
    case = CASES[name]
    gen = _generator_module(data_dir)
    clouds = []
    fn_name, target_kwargs, source_kwargs = case["generator"]
    for pcd, kwargs in ((case["target_pcd"], target_kwargs),
                        (case["source_pcd"], source_kwargs)):
        path = os.path.join(data_dir, pcd)
        if os.path.exists(path):
            clouds.append(gen.load_pcd(path))
        else:
            print(f"[demo] {path} not found - regenerating in memory")
            clouds.append(getattr(gen, fn_name)(**kwargs))
    return clouds[0], clouds[1]


def pose_error(T_est, T_true):
    """Per-DOF error [tx, ty, tz, roll, pitch, yaw] of T_est vs T_true."""
    D = np.linalg.inv(T_true) @ T_est
    R, t = D[:3, :3], D[:3, 3]
    rot = 0.5 * np.array([R[2, 1] - R[1, 2],
                          R[0, 2] - R[2, 0],
                          R[1, 0] - R[0, 1]])
    return np.concatenate([t, rot])


def dominant_dof(eigenvector):
    return DOF_LABELS[int(np.argmax(np.abs(eigenvector)))]


def run_case(name, data_dir):
    case = CASES[name]
    target, source_surface = load_case(name, data_dir)
    T_true = makeT(expSO3(case["rvec"]), case["tvec"])
    scan = transform_points(np.linalg.inv(T_true), source_surface)
    scan32 = scan.astype(np.float32)

    results = {}
    for mode, kwargs in (
            ("baseline", dict(lm_damping=True)),
            ("sr", dict(use_solution_remapping=True, lm_damping=True))):
        engine = PlaneICP(max_iter=50, max_dist=1.0, tol=1e-6, k=10)
        engine.set_target(target)
        T = engine.align(scan32, **kwargs)
        deg = analyse_hessian(engine.last_hessian.astype(float))
        results[mode] = {
            "T": T,
            "deg": deg,
            "errors": np.abs(pose_error(T, T_true)),
            "aligned": transform_points(T, scan),
        }
    return {
        "name": name,
        "case": case,
        "target": target,
        "scan": scan,
        "T_true": T_true,
        "results": results,
    }


def print_report(out):
    case = out["case"]
    print(f"\n=== {case['title']} ===")
    deg = out["results"]["sr"]["deg"]
    print(f"eigenvalues      : {np.array2string(deg.eigenvalues, precision=2)}")
    print(f"condition number : {deg.condition_number:.1f}")
    print(f"SR threshold     : {deg.lambda_threshold:.1f}")
    print(f"constrained DOFs : {deg.num_constrained_dof}/6")
    flags = ", ".join(
        f"{dominant_dof(deg.eigenvectors[:, i])} (lam={deg.eigenvalues[i]:.1f})"
        for i in range(6) if deg.degenerate_mask[i])
    print(f"degenerate       : {flags or 'none'}")

    print(f"\n{'DOF':<6} {'no SR':>12} {'with SR':>12}")
    base = out["results"]["baseline"]["errors"]
    sr = out["results"]["sr"]["errors"]
    for i, label in enumerate(DOF_LABELS):
        unit = "m" if i < 3 else "rad"
        print(f"{label:<6} {base[i]:>10.4f} {unit} {sr[i]:>10.4f} {unit}")


def make_figure(out, plt):
    case = out["case"]
    target = out["target"]
    scan = out["scan"]
    base = out["results"]["baseline"]
    sr = out["results"]["sr"]
    deg = sr["deg"]

    fig = plt.figure(figsize=(12.5, 9.5), constrained_layout=True)
    fig.suptitle(case["title"], fontsize=14)

    # A: before alignment (3D).
    ax_a = fig.add_subplot(2, 2, 1, projection="3d")
    step = max(1, target.shape[0] // 1500)
    ax_a.scatter(*target[::step].T, s=2, c=COLORS["target"], alpha=0.35,
                 label="target scan")
    ax_a.scatter(*scan[::step].T, s=2, c=COLORS["initial"], alpha=0.6,
                 label="source scan, initial pose")
    ax_a.set_title("A - before alignment")
    ax_a.set_xlabel("x [m]"); ax_a.set_ylabel("y [m]"); ax_a.set_zlabel("z [m]")
    ax_a.view_init(elev=10, azim=-88)  # look down the stair width: step profile
    ax_a.legend(loc="upper left", fontsize=8, markerscale=4)

    # B: after alignment, top-down, with a zoom inset on the drift.
    ax_b = fig.add_subplot(2, 2, 2)
    ax_b.scatter(target[:, 0], target[:, 1], s=2, c=COLORS["target"],
                 alpha=0.3, label="target")
    ax_b.scatter(base["aligned"][:, 0], base["aligned"][:, 1], s=2,
                 c=COLORS["baseline"], alpha=0.4, label="aligned, no SR")
    ax_b.scatter(sr["aligned"][:, 0], sr["aligned"][:, 1], s=2,
                 c=COLORS["sr"], alpha=0.4, label="aligned, with SR")
    ax_b.set_title("B - after alignment (top-down)")
    ax_b.set_xlabel("x [m]"); ax_b.set_ylabel("y [m]")
    ax_b.set_aspect("equal")
    ax_b.legend(loc="upper left", fontsize=8, markerscale=4)

    # Zoom anchored to the +x/+y extreme of the target so the lateral
    # drift of the vermillion cloud is visible at true scale.
    anchor = target[np.argmax(target[:, 0] + target[:, 1])][:2]
    half = 0.45
    axins = ax_b.inset_axes([0.58, 0.05, 0.4, 0.4])
    for pts, color in ((target, COLORS["target"]),
                       (base["aligned"], COLORS["baseline"]),
                       (sr["aligned"], COLORS["sr"])):
        m = (np.abs(pts[:, 0] - anchor[0]) < half) & \
            (np.abs(pts[:, 1] - anchor[1]) < half)
        axins.scatter(pts[m, 0], pts[m, 1], s=4, c=color, alpha=0.5)
    axins.set_xlim(anchor[0] - half, anchor[0] + half)
    axins.set_ylim(anchor[1] - half, anchor[1] + half)
    axins.set_xticks([]); axins.set_yticks([])
    ax_b.indicate_inset_zoom(axins, edgecolor="black")

    # C: eigenvalue spectrum vs SR threshold.
    ax_c = fig.add_subplot(2, 2, 3)
    colors = [COLORS["baseline"] if deg.degenerate_mask[i] else COLORS["sr"]
              for i in range(6)]
    eig_floor = max(deg.eigenvalues.min(), 1e-3)
    ax_c.bar(range(6), np.maximum(deg.eigenvalues, eig_floor), color=colors,
             width=0.6)
    ax_c.axhline(deg.lambda_threshold, color="black", linestyle="--",
                 linewidth=1.2)
    ax_c.text(0.02, deg.lambda_threshold * 1.15,
              "SR threshold  sqrt(lam_max/lam_min)", fontsize=8)
    ax_c.set_yscale("log")
    ax_c.set_xticks(range(6))
    ax_c.set_xticklabels(
        [dominant_dof(deg.eigenvectors[:, i]) for i in range(6)], fontsize=9)
    ax_c.set_xlabel("eigen-direction (dominant DOF)")
    ax_c.set_ylabel("eigenvalue")
    ax_c.set_title(
        f"C - Hessian spectrum - {deg.num_constrained_dof}/6 constrained "
        f"(degenerate in vermillion)")

    # D: per-DOF error vs ground truth.
    ax_d = fig.add_subplot(2, 2, 4)
    x = np.arange(6)
    err_floor = 1e-6
    ax_d.bar(x - 0.18, np.maximum(base["errors"], err_floor), width=0.36,
             color=COLORS["baseline"], label="no SR")
    ax_d.bar(x + 0.18, np.maximum(sr["errors"], err_floor), width=0.36,
             color=COLORS["sr"], label="with SR")
    ax_d.set_yscale("log")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(DOF_LABELS, fontsize=9)
    ax_d.set_ylabel("|error| vs ground truth  [m or rad]")
    ax_d.set_title("D - final pose error per DOF")
    ax_d.legend(fontsize=8)

    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Degeneracy detection + solution remapping demo.")
    parser.add_argument("--case", choices=["stair", "plane", "both"],
                        default="both")
    parser.add_argument("--save", action="store_true",
                        help="write imgs/degeneracy_*.png instead of showing")
    parser.add_argument("--show", action="store_true",
                        help="open interactive windows (default)")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)

    names = ["stair", "plane"] if args.case == "both" else [args.case]
    outputs = [run_case(name, args.data_dir) for name in names]
    for out in outputs:
        print_report(out)

    try:
        if args.save and not args.show:
            # Select a headless backend; must happen before pyplot loads.
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[demo] matplotlib is not installed - skipping the figures "
              "(pip install matplotlib)")
        return 0

    for out in outputs:
        fig = make_figure(out, plt)
        if args.save:
            os.makedirs(os.path.join(REPO_DIR, "imgs"), exist_ok=True)
            path = os.path.join(REPO_DIR, "imgs", out["case"]["png"])
            fig.savefig(path, dpi=110)
            print(f"[demo] wrote {path}")
    if args.show or not args.save:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
