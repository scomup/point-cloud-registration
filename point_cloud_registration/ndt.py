import numpy as np
from point_cloud_registration.registration import Registration
from point_cloud_registration.voxel import VoxelGrid
from point_cloud_registration.math_tools import skews, transform_points, plus, skew


class NDT(Registration):
    def __init__(self, voxel_size=0.5, max_iter=100, max_dist=2, tol=1e-6):
        super().__init__(max_iter=max_iter, tol=tol)
        self.voxels = None
        self.voxel_size = voxel_size
        self.max_dist = max_dist

    def update_target(self, target):
        if self.voxels is None:
            self.voxels = VoxelGrid(self.voxel_size)
        self.voxels.add_points(target)

    def set_target(self, target):
        self.voxels = None
        self.update_target(target)

    def calc_H_g_e2(self, cur_T, source):
        if self.voxels is None:
            raise ValueError("Target is not set.")

        R = cur_T[:3, :3]
        src_trans = transform_points(cur_T, source)

        # Query voxels (icov and mean)
        icov, qs = self.voxels.query(src_trans.astype(np.float32), ['icov', 'mean'])
        icov = np.array(icov)
        qs = np.array(qs)

        # Compute residuals (d = src_trans - qs)
        d = src_trans - qs  # shape: (N, 3)

        # Construct Jacobian (J)
        J = np.zeros((source.shape[0], 3, 6))  # shape: (N, 3, 6)
        J[:, :, :3] = np.eye(3)  # df/dt = I
        J[:, :, 3:] = -R @ skews(source) # df/dr = -R @ skew(src)

        # Compute H (Hessian) = Σ(Jᵀ @ icov @ J)
        H = np.einsum('nji,njk,nkl->il', J, icov, J)  # shape: (6, 6)

        # Compute g (gradient) = Σ(Jᵀ @ icov @ d)
        g = np.einsum('nji,njk,nk->i', J, icov, d)  # shape: (6,)

        # Compute e2 (error squared) = Σ(dᵀ @ icov @ d)
        e2 = np.einsum('ni,nij,nj->', d, icov, d)  # scalar

        return H, g, e2
