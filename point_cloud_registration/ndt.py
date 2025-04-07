"""
Copyright 2025 Liu Yang
Distributed under MIT license. See LICENSE for more information.
"""

import numpy as np
from point_cloud_registration.registration import Registration
from point_cloud_registration.voxel import VoxelGrid
from point_cloud_registration.math_tools import skews, transform_points, plus, skew


class NDT(Registration):
    def __init__(self, voxel_size=1.0, max_iter=30, max_dist=2, tol=1e-3):
        super().__init__(max_iter=max_iter, tol=tol)
        self.voxel_size = voxel_size
        self.max_dist = max_dist

    def set_target(self, target):
        self.voxels = VoxelGrid(self.voxel_size)
        self.voxels.set_points(target)
        self.voxels.calc_icov() # need for ndt
        self._is_target_set = True

    def calc_H_g_e2(self, cur_T, source):
        R = cur_T[:3, :3]
        src_trans = transform_points(cur_T.astype(np.float32), source)

        # Query voxels (icov and mean)
        query_data = self.voxels.query(src_trans, ['icov', 'mean'])
        icov = query_data['icov']
        means = query_data['mean']
        dist = query_data['dist']
        mask = dist < self.max_dist
        means = means[mask]
        icov = icov[mask]
        src_mask = source[mask]
        src_trans = src_trans[mask]

        diff = src_trans - means  # shape: (N, 3)

        # J0 = np.eye(3)  # 3x3
        J1 = -R @ skews(src_mask) # shape: (N, 3, 3)
        J0T_icov = icov
        J1T_icov = np.einsum('kji,kjl->kil', J1, icov)

        # only upper triangle version,
        # equal to np.einsum('nji,njk,nkl->il', J, icov, J) 
        # but faster than np.einsum
        H_00 = np.sum(J0T_icov, axis=0) # sum (J0.T * icov * J0)
        H_01  = np.sum(np.transpose(J1T_icov, (0, 2, 1)),axis=0)
        H_11 = np.einsum('nij,njk->ik', J1T_icov, J1)
        H = np.zeros((6, 6))
        H[:3, :3] = H_00
        H[:3, 3:] = H_01
        H[3:, :3] = H_01.T
        H[3:, 3:] = H_11

        g0 = np.einsum('nij,nj->i', J0T_icov, diff)
        g1 = np.einsum('nij,nj->i', J1T_icov, diff)
        g = np.hstack([g0, g1])  # shape: (6,)

        e2 = np.einsum('ni,nij,nj->', diff, icov, diff)  # scalar

        return H, g, e2


    def calc_H_g_e2_no_parallel_ver(self, cur_T, source):
        # calc_H_g_e2_no_parallel_ver
        """
        Note: This is a non-parallel version of calc_H_g_e2.
        This function is just for helping to understand the algorithm.
        the logic is the totally same as calc_H_g_e2.
        """
        R = cur_T[:3, :3]
        src_trans = transform_points(cur_T.astype(np.float32), source)

        # Query voxels (icov and mean)
        query_data = self.voxels.query(src_trans, ['icov', 'mean'])
        icov = query_data['icov']
        means = query_data['mean']
        dist = query_data['dist']
        mask = dist < self.max_dist
        means = means[mask]
        icov = icov[mask]
        #src_mask = source[mask]
        src_trans = src_trans[mask]
        H = np.zeros((6, 6))
        g = np.zeros(6)
        e2 = 0

        for i in range(source.shape[0]):
            J = np.zeros((3, 6))
            # Jacobian of the transformation
            J[:, :3] = np.eye(3)
            # Jacobian of the rotation
            J[:, 3:] = -R @ skew(source[i])
            # residual
            r = src_trans[i] - means[i]

            if dist[i] > self.max_dist:
                continue
            H += J.T @ icov[i] @  J
            g += J.T @ icov[i] @ r
            e2 += r @ icov[i] @ r
        return H, g, e2