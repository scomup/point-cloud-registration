"""
Copyright 2025 Liu Yang
Distributed under MIT license. See LICENSE for more information.
"""

import numpy as np
from point_cloud_registration.registration import Registration
from point_cloud_registration.math_tools import skews, transform_points, skew2, skew
from point_cloud_registration.kdtree import KDTree


class ICP(Registration):
    def __init__(self, max_iter=30, max_dist=2, tol=1e-3):
        super().__init__(max_iter=max_iter, tol=tol)
        self.max_dist = max_dist

    def set_target(self, target):
        # target = target.astype(np.float32)
        target = target.astype(np.float32)
        self.kdtree = KDTree(target)
        self.target = target
        self._is_target_set = True

    def calc_H_g_e2(self, cur_T, source):
        """
        Calculate the Hessian, gradient, and squared error.
        This function is heavily optimized for speed.
        :param cur_T: Current transformation (4x4 array).
        :param source: Source point cloud (Nx3 array).
        :return: Hessian (6x6 array), gradient (6 array), squared error (scalar).
        """
        src_trans = transform_points(cur_T.astype(np.float32), source)
        dist, idx = self.kdtree.query(src_trans)
        mask = dist < self.max_dist
        idx = idx[mask]
        src_trans = src_trans[mask]
        num = src_trans.shape[0]
        src_mask = source[mask]
        qs = self.target[idx]
        rs = src_trans - qs
        R = cur_T[:3, :3]
        S = skews(src_mask)
        S_sum = skew(np.sum(src_mask, axis=0))
        H_ll = num * np.eye(3)
        H_lr = - R @ S_sum
        H_rr = skew2(src_mask)
        H = np.zeros((6, 6))
        H[:3, :3] = H_ll
        H[:3, 3:] = H_lr
        H[3:, :3] = H_lr.T
        H[3:, 3:] = H_rr
        g0 = rs.sum(axis=0)
        Rt_r = rs @ R.T
        g1 = np.einsum('nij,ni->j', S, -Rt_r)
        g = np.hstack([g0, g1])
        e2 = np.sum(rs * rs)
        return H, g, e2

    def calc_H_g_e2_no_parallel_ver(self, cur_T, source):
        """
        Note: This is a non-parallel version of calc_H_g_e2.
        This function is just for helping to understand the algorithm.
        the logic is the totally same as calc_H_g_e2.
        """
        src_trans = transform_points(cur_T, source)
        dist, idx = self.kdtree.query(src_trans.astype(np.float32))
        mask = dist < self.max_dist
        src_trans = src_trans[mask]
        num = src_trans.shape[0]
        # Find corresponding target points
        qs = self.target[idx]
        R = cur_T[:3, :3]
        H = np.zeros((6, 6))
        g = np.zeros(6)
        e2 = 0
        for i in range(num):
            J = np.zeros((3, 6))
            # Jacobian of the transformation
            J[:, :3] = np.eye(3)
            # Jacobian of the rotation
            J[:, 3:] = -R @ skew(source[i])
            # residual
            r = src_trans[i] - qs[i]
            # Hessian
            H += J.T @ J
            # Gradient
            g += J.T @ r
            # Squared error
            e2 += r @ r
        return H, g, e2
