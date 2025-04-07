"""
Copyright 2025 Liu Yang
Distributed under MIT license. See LICENSE for more information.

TODO (liu): Try using Caratheodory's theorem to select a coreset of points to speed up VPlaneICP or PlaneICP.
This file is my Python implementation of Caratheodory's theorem-based algorithms for exact point 
cloud downsampling.

Current problems:
1. Only works when the error is very small.
2. May not work for extracted point sets with small sizes.
"""

import numpy as np
import q3dviewer as q3d
from voxel import VoxelGrid
from caratheodory import fast_caratheodory, create_gn_set
from point_cloud_registration.voxelized_plane_icp import skews
from math_tools import makeRt, expSO3, makeT


class FastVPlaneICP:
    def __init__(self, voxel_size, max_iter=30, max_dist=2, tol=1e-3, N_target=1024, debug=True):
        self.num = 0
        self.voxels = VoxelGrid(voxel_size)
        self.max_iter = max_iter
        self.tol = tol
        self.max_dist = max_dist
        self.N_target = N_target
        self.debug = debug

    def get_coreset(self, Js, rs, ws, N_target):
        P = create_gn_set(Js, rs)
        _, w, indices = fast_caratheodory(P, ws, 64, N_target)
        return indices, w

    def set_target(self, points):
        self.voxels.set_points(points)

    def linearize(self, cur_T, source):
            R, t = makeRt(cur_T)
            source_trans = (R @ source.T).T + t
            dist, idx = self.voxels.kdtree.query(source_trans)
            Js = np.zeros([source.shape[0], 6])
            # Find corresponding target points
            means = np.array([self.voxels.voxels[self.voxels.voxel_keys[i]].mean for i in idx])
            norms = np.array([self.voxels.voxels[self.voxels.voxel_keys[i]].norm for i in idx])
            # Compute transformation
            pw = source_trans
            rs = np.einsum('ij,ij->i', norms, pw - means)
            Js[:, :3] = norms
            Js[:, 3:] = np.einsum('ijk,ki->ij', skews(source), R.T @ norms.T)
            w = np.ones(source.shape[0])
            return Js, rs, w

    def align(self, source, init_T=np.eye(4)):
        cur_T = init_T.copy()
        using_coreset = False
        Js = None
        rs = None
        ws = None
        indices = None
        coreset_moving_th = 1e-2
        for i in range(self.max_iter):

            if using_coreset:
                source_selected = source[indices]
                Js, rs, _ = self.linearize(cur_T, source_selected)
            else:
                Js, rs, ws = self.linearize(cur_T, source)

            # gauss-newton step
            H = Js.T @ (ws[:, np.newaxis] * Js)
            g = Js.T @ (ws * rs)
            e2 = rs.T @ (ws * rs)

            if self.debug:
                print(f"iter {i}, points size {len(rs)}, error {e2}")

            dx = -np.linalg.solve(H, g)

            moving = np.linalg.norm(dx)

            # Update transformation
            dR = expSO3(dx[3:])
            dt = dx[:3]
            dT = makeT(dR, dt)
            if moving < self.tol:
                break

            if moving < coreset_moving_th:
                indices, ws = self.get_coreset(Js, rs, ws, self.N_target)
                using_coreset = True
            #else:
            #    using_coreset = False

            cur_T = cur_T @ dT

        return cur_T

