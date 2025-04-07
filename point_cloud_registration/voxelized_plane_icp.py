"""
Copyright 2025 Liu Yang
Distributed under MIT license. See LICENSE for more information.
"""

import numpy as np
from point_cloud_registration.registration import Registration
from point_cloud_registration.voxel import VoxelGrid
from point_cloud_registration.math_tools import skews, transform_points, skew, skew_time_vector


class VPlaneICP(Registration):
    def __init__(self, voxel_size=1.0, max_iter=30, max_dist=2, tol=1e-3):
        super().__init__(max_iter=max_iter, tol=tol)
        self.voxel_size = voxel_size
        self.max_dist = max_dist

    def set_target(self, target):
        self.voxels = VoxelGrid(self.voxel_size)
        self.voxels.set_points(target)
        self._is_target_set = True

    def calc_H_g_e2(self, cur_T, source):
        """
        Calculate the Hessian, gradient, and squared error.
        This function is heavily optimized for speed.
        :param cur_T: Current transformation (4x4 array).
        :param source: Source point cloud (Nx3 array).
        :return: Hessian (6x6 array), gradient (6 array), squared error (scalar).
        """
        R = cur_T[:3, :3]
        src_trans = transform_points(cur_T.astype(np.float32), source)

        query_data = self.voxels.query(
            src_trans, ['mean', 'norm'])
        
        dist = query_data['dist']
        mask = dist < self.max_dist
        means = query_data['mean'][mask]
        norms = query_data['norm'][mask]
        src_trans = src_trans[mask]
        diff = src_trans - means
        src_mask = source[mask]
        rs = np.einsum('ij,ij->i', norms, diff)
        Jt = norms
        Rt_norms = R.T @ norms.T
        # # equal to skew_time_vector
        # Jr = np.einsum('ijk,ki->ij', skews(src_mask), Rt_norms) 
        Jr = skew_time_vector(src_mask, Rt_norms.T)
        H_ll = np.einsum('ij,ik->jk', Jt, Jt)
        H_lr = np.einsum('ij,ik->jk', Jt, Jr)
        H_rr = np.einsum('ij,ik->jk', Jr, Jr)
        H = np.zeros((6, 6))
        H[:3, :3] = H_ll
        H[:3, 3:] = H_lr
        H[3:, :3] = H_lr.T
        H[3:, 3:] = H_rr
        g0 = np.sum(Jt * rs[:,np.newaxis],axis=0)
        g1 = np.sum(Jr * rs[:,np.newaxis],axis=0)

        g = g = np.hstack([g0, g1])
        e2 = np.sum(rs * rs)
        # t3 = time.time()
        return H, g, e2


    def calc_H_g_e2_no_parallel_ver(self, cur_T, source):
        """
        Note: This is a non-parallel version of calc_H_g_e2.
        This function is just for helping to understand the algorithm.
        the logic is the totally same as calc_H_g_e2.
        """

        if self.is_target_set() is False:
            raise ValueError("Target is not set.")
        R = cur_T[:3, :3]
        src_trans = transform_points(cur_T.astype(np.float32), source)
        query_data = self.voxels.query(
            src_trans, ['mean', 'norm'])

        dist = query_data['dist']
        mask = dist < self.max_dist
        means = query_data['mean'][mask]
        norms = query_data['norm'][mask]
        src_trans = src_trans[mask]

        H = np.zeros((6, 6))
        g = np.zeros(6)
        e2 = 0
        for i in range(source.shape[0]):
            n = norms[i]
            r = n @ (src_trans[i] - means[i])
            J = np.zeros((1, 6))
            J[0, :3] = n
            J[0, 3:] = skew(source[i]) @ (R.T @ n.T)
            if np.abs(r) > self.max_dist:
                continue
            H += J.T @ J
            g += J[0] * r
            e2 += r * r
        return H, g, e2