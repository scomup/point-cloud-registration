import numpy as np
from point_cloud_registration.voxel import VoxelGrid
from point_cloud_registration.math_tools import makeRt, expSO3, makeT, skews


class VoxelPoint2PlaneICP:
    def __init__(self, voxel_size=0.5, max_iter=100, max_dist=2, tol=1e-6):
        self.num = 0
        self.voxels = VoxelGrid(voxel_size)
        self.max_iter = max_iter
        self.tol = tol
        self.max_dist = max_dist

    def update_target(self, points):
        self.voxels.add_points(points)

    def linearize(self, cur_T, source):
            R, t = makeRt(cur_T)
            source_trans = (R @ source.T).T + t
            dist, idx = self.voxels.kdtree.query(source_trans)
            good_idx = dist < self.max_dist
            idx = idx[good_idx]
            source_orig = source[good_idx].copy()
            Js = np.zeros([source_orig.shape[0], 6])
            # Find corresponding target points
            means = np.array([self.voxels.voxels[self.voxels.voxel_keys[i]].mean for i in idx])
            norms = np.array([self.voxels.voxels[self.voxels.voxel_keys[i]].norm for i in idx])
            # Compute transformation
            pw = source_trans[good_idx]
            rs = np.einsum('ij,ij->i', norms, pw - means)
            Js[:, :3] = norms
            Js[:, 3:] = np.einsum('ijk,ki->ij', skews(source_orig), R.T @ norms.T)
            return Js, rs, good_idx

    def fit(self, source, init_T=np.eye(4), verbose=False):
        cur_T = init_T.copy()
        for i in range(self.max_iter):
            Js, rs, _ = self.linearize(cur_T, source)
            # gauss-newton step
            H = Js.T @ Js
            g = Js.T @ rs
            e2 = rs.T @ rs
            if verbose:
                 print(f"iter {i}, error {e2}")
            dx = -np.linalg.solve(H, g)
            # Update transformation
            dR = expSO3(dx[3:])
            dt = dx[:3]
            dT = makeT(dR, dt)
            if np.linalg.norm(dx) < self.tol:
                break
            cur_T = cur_T @ dT

        return cur_T
