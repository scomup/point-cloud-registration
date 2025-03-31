import numpy as np
import q3dviewer as q3d
from voxel import VoxelGrid
from caratheodory import fast_caratheodory, create_gn_set
from point_cloud_registration.voxelized_point_plane_icp import skews
from math_tools import makeRt, expSO3, makeT


class FastVoxelPoint2PlaneICP:
    def __init__(self, voxel_size, max_iter=100, max_dist=2, tol=1e-6, N_target=1024, debug=True):
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

    def update_target(self, points):
        self.voxels.add_points(points)

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

if __name__ == '__main__':
    # Generate N x 3 points
    map, _ = q3d.load_pcd("/home/liu/tmp/recorded_frames/clouds/0.pcd")
    scan, _ = q3d.load_pcd("/home/liu/tmp/recorded_frames/clouds/2.pcd")
    map = map['xyz']
    scan = scan['xyz']


    icp = FastVoxelPoint2PlaneICP(0.5, N_target=2000)
    icp.update_target(map)
    T_new = icp.align(scan, init_T=np.eye(4))
    R_new, t_new = makeRt(T_new)
    scan_new = (R_new @ scan.T).T + t_new

    app = q3d.QApplication([])

    # create viewer
    viewer = q3d.Viewer(name='example')
    # add all items to viewer
    viewer.add_items({
        'grid': q3d.GridItem(size=10, spacing=1),
        'map': q3d.CloudItem(size=0.005, alpha=0.5, point_type='SPHERE', \
                              color_mode='FLAT', color='#00ff00', depth_test=True),
        'scan': q3d.CloudItem(size=0.005, alpha=0.5, point_type='SPHERE', \
                              color_mode='FLAT', color='#ff0000', depth_test=True),
        'norm': q3d.LineItem(width=2, color='#00ff00', line_type='LINES')})

    viewer['map'].set_data(map)
    viewer['scan'].set_data(scan_new)

    viewer.show()
    app.exec()

