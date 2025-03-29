import numpy as np
from point_cloud_registration.registration import Registration
from point_cloud_registration.voxel import VoxelGrid
from point_cloud_registration.math_tools import makeRt, expSO3, makeT, skews, huber_weight


class VoxelizedPoint2PlaneICP(Registration):
    def __init__(self, voxel_size=0.5, max_iter=100, max_dist=2, tol=1e-6):
        super().__init__(max_iter=max_iter, tol=tol)
        self.voxels = None
        self.voxel_size = voxel_size
        self.max_dist = max_dist

    def update_target(self, target):
        if self.voxels is None:
            self.voxels = VoxelGrid(self.voxel_size)
        self.voxels.add_points(target)

    def linearize(self, cur_T, source):
            R, t = makeRt(cur_T)
            source_trans = (R @ source.T).T + t
           