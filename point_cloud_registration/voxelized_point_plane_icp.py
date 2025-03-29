import numpy as np
from point_cloud_registration.registration import Registration
from point_cloud_registration.voxel import VoxelGrid
from point_cloud_registration.math_tools import makeRt, plus, skews, transform_points


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

    def set_target(self, target):
        self.voxels = None
        self.update_target(target)

    def linearize(self, cur_T, source):
        R = cur_T[:3, :3]
        source_trans = transform_points(cur_T, source)
        dist, idx = self.voxels.kdtree.query(
            source_trans.astype(np.float32))
        Js = np.zeros([source.shape[0], 1, 6])
        # Find corresponding target points
        means = np.array(
            [self.voxels.voxels[self.voxels.voxel_keys[i]].mean for i in idx])
        norms = np.array(
            [self.voxels.voxels[self.voxels.voxel_keys[i]].norm for i in idx])
        # Compute transformation
        rs = np.einsum('ij,ij->i', norms, source_trans - means)[:, np.newaxis]
        Js[:, 0, :3] = norms
        Js[:, 0, 3:] = np.einsum('ijk,ki->ij', skews(source),
                                 R.T @ norms.T)
        weights = np.ones(source.shape[0])
        weights[dist > self.max_dist] = 0
        return Js, rs, weights

