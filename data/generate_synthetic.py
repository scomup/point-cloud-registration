#!/usr/bin/env python3
"""
Deterministic generators for the synthetic degenerate test clouds.

Two scenes, each under-constraining registration in a known way:

* staircase — treads (normals +z) and risers (normals +x) constrain
  tx, tz and all three rotations, but nothing observes translation
  across the stair width: exactly one degenerate DOF (ty).
* plane — a featureless flat seafloor. Only tz, roll and pitch are
  observable: three degenerate DOFs (tx, ty, yaw).

Regeneration is byte-identical (fixed seeds, fixed format string), so
the committed .pcd files can always be reproduced with

    python3 data/generate_synthetic.py

Design notes that carry the demo's story:

* Faces keep a margin (~ the k-NN neighborhood radius) away from the
  concave tread/riser folds. Without it, PCA normal estimation blends
  the two faces at the fold, the blended normals pick up spurious
  y-components, and the ty direction stops being flagged as degenerate.
* Centering is deterministic (never the empirical mean), so clouds
  sampled with different seeds lie on the *same* surface — required for
  target/source pairs that emulate two scans of one scene.

NumPy only; no external point-cloud dependencies.
"""
import argparse
import os

import numpy as np


def make_staircase(n_steps=6, tread=1.0, riser=0.5, width=6.0,
                   pts_per_face=220, margin=0.10, noise=0.01, seed=42):
    """
    Point cloud of a staircase marching along +x and rising in +z.

    Step k contributes a tread (horizontal, normal +z) at height
    (k+1)*riser and a riser (vertical, normal +x) at x = k*tread.
    Returns an (N, 3) float64 array centered deterministically.
    """
    rng = np.random.default_rng(seed)
    faces = []
    for k in range(n_steps):
        # Tread: z = (k+1)*riser, x in [k*tread+margin, (k+1)*tread-margin].
        u = rng.uniform(k * tread + margin, (k + 1) * tread - margin,
                        pts_per_face)
        v = rng.uniform(0.0, width, pts_per_face)
        tread_face = np.column_stack([
            u, v, np.full(pts_per_face, (k + 1) * riser)])
        # Riser: x = k*tread, z in [k*riser+margin, (k+1)*riser-margin].
        w = rng.uniform(k * riser + margin, (k + 1) * riser - margin,
                        pts_per_face)
        v2 = rng.uniform(0.0, width, pts_per_face)
        riser_face = np.column_stack([
            np.full(pts_per_face, k * tread), v2, w])
        faces.extend([tread_face, riser_face])
    points = np.vstack(faces)
    points += rng.normal(0.0, noise, points.shape)
    center = np.array([n_steps * tread / 2.0, width / 2.0,
                       n_steps * riser / 2.0])
    return points - center


def make_plane(half_extent=6.0, num_points=2500, noise=0.01, seed=7):
    """
    A featureless flat seafloor: uniform samples of z = 0 over
    [-half_extent, half_extent]^2 with Gaussian surface noise.
    """
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-half_extent, half_extent, (num_points, 2))
    z = rng.normal(0.0, noise, num_points)
    return np.column_stack([xy, z])


def save_pcd(path, points):
    """
    Write an Nx3 array as an ASCII PCD v0.7 file (x y z).

    Coordinates are stored at float32 / six-decimal precision — the PCD
    FIELDS declare SIZE 4 — so a save/load roundtrip is exact to ~1e-6,
    not bit-exact against float64 input. Regeneration from the fixed
    seeds is byte-identical because the quantization itself is
    deterministic.
    """
    points = np.asarray(points, dtype=np.float32)
    n = points.shape[0]
    header = "\n".join([
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS x y z",
        "SIZE 4 4 4",
        "TYPE F F F",
        "COUNT 1 1 1",
        f"WIDTH {n}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {n}",
        "DATA ascii",
    ])
    with open(path, "w") as f:
        f.write(header + "\n")
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def load_pcd(path):
    """Read an ASCII PCD file with x y z fields into an (N, 3) array."""
    points = []
    with open(path) as f:
        in_data = False
        for line in f:
            if in_data:
                parts = line.split()
                if len(parts) >= 3:
                    points.append([float(parts[0]), float(parts[1]),
                                   float(parts[2])])
            elif line.startswith("DATA"):
                if line.split()[1] != "ascii":
                    raise ValueError(f"{path}: only ASCII PCD is supported")
                in_data = True
    return np.array(points)


# (filename, generator, kwargs) — target/source pairs are independent
# samplings of the same surface, so registration sees realistic
# correspondence noise instead of a permutation of identical points.
CLOUDS = [
    ("synthetic_staircase_target.pcd", make_staircase,
     dict(seed=42, pts_per_face=220)),
    ("synthetic_staircase_source.pcd", make_staircase,
     dict(seed=2024, pts_per_face=220)),
    ("synthetic_plane_target.pcd", make_plane,
     dict(seed=7, num_points=2500)),
    ("synthetic_plane_source.pcd", make_plane,
     dict(seed=99, num_points=2000)),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out-dir",
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help="Directory to write the .pcd files into "
                             "(default: this file's directory).")
    args = parser.parse_args()
    for name, generator, kwargs in CLOUDS:
        path = os.path.join(args.out_dir, name)
        points = generator(**kwargs)
        save_pcd(path, points)
        print(f"wrote {path} ({points.shape[0]} points)")


if __name__ == "__main__":
    main()
