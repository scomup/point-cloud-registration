import numpy as np
from point_cloud_registration.registration import Registration
from point_cloud_registration.math_tools import skews, transform_points
from point_cloud_registration.kdtree import KDTree

class ICP(Registration):
    def __init__(self, max_iter=100, max_dist=2, tol=1e-6):
        super().__init__(max_iter=max_iter, tol=tol)
        self.max_dist = max_dist
        self.kdtree = None

    def set_target(self, target):
        self.kdtree = KDTree(target)
        self.target = target

    def linearize(self, cur_T, source):
            if self.kdtree is None:
                raise ValueError("Target is not set.")
            src_trans = transform_points(cur_T, source)
            _, idx = self.kdtree.query(src_trans.astype(np.float32))
            idx = idx
            Js = np.zeros([source.shape[0], 6])
            # Find corresponding target points
            qs = self.target[idx]
            # Compute transformation

            num = src_trans.shape[0]
            Js = np.zeros([num, 3, 6])
            rs = np.zeros([num, 3])

            # the residual of icp
            rs = src_trans - qs

            # the Jacobian of icp
            R = cur_T[:3, :3]
            Js[:, :, :3] = np.repeat(np.eye(3)[np.newaxis, :, :], num, axis=0)
            Js[:, :, 3:] = -R @ skews(source)

            weights = np.ones(num)
            weights[np.linalg.norm(rs, axis=1) > self.max_dist] = 0
            return Js, rs, weights



