import numpy as np
from point_cloud_registration.registration import Registration
from point_cloud_registration.math_tools import makeRt, expSO3, makeT, skews, plus
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
            R, t = makeRt(cur_T)
            src_trans = (R @ source.T).T + t
            dist, idx = self.kdtree.query(src_trans.astype(np.float32))
            idx = idx
            Js = np.zeros([source.shape[0], 6])
            # Find corresponding target points
            qs = self.target[idx]
            # Compute transformation

            num = src_trans.shape[0]
            Js = np.zeros([num, 3, 6])
            rs = np.zeros([num, 3])
            rs = src_trans - qs
            Js[:, :, :3] = np.repeat(np.eye(3)[np.newaxis, :, :], num, axis=0)
            Js[:, :, 3:] = -R @ skews(source)

            weights = np.ones(num)
            weights[np.linalg.norm(rs) > self.max_dist] = 0
            return Js, rs, weights

    def fit(self, source, init_T=np.eye(4), verbose=False):
        if self.kdtree is None:
            raise ValueError("Target is not set.")
        cur_T = init_T.copy()
        for i in range(self.max_iter):
            Js, rs, _ = self.linearize(cur_T, source)
            # Gauss-Newton step

            JsT = Js.transpose(0, 2, 1)
            H = np.einsum('nij,njk->ik', JsT, Js)
            g = np.einsum('nij,nj->i', JsT, rs)
            e2 = np.sum(rs ** 2)
            
            if verbose:
                print(f"iter {i}, error {e2}")
            
            dx = -np.linalg.solve(H, g)
            # Update transformation
            if np.linalg.norm(dx) < self.tol:
                break

            cur_T = plus(cur_T, dx)

        return cur_T


if __name__ == '__main__':
    import q3dviewer as q3d

    # Generate N x 3 points
    map, _ = q3d.load_pcd("/home/liu/tmp/recorded_frames/clouds/0.pcd")
    scan, _ = q3d.load_pcd("/home/liu/tmp/recorded_frames/clouds/2.pcd")
    map = map['xyz']
    scan = scan['xyz']
    T = np.array([[0.91724685,  0.39554969, -0.04691057, -0.01918991],
                  [-0.39801572,  0.90555297, -0.14683008, -0.02283431],
                  [-0.01559759,  0.15334987,  0.98804928, -0.00580279],
                  [0.,  0.,  0.,  1.]])

    # T = np.eye(4)
    icp = ICP(voxel_size=0.3, max_iter=100, max_dist=0.1, tol=1e-5)
    icp.update_target(map)
    T_new = icp.fit(scan, init_T=T, verbose=True)
    R_new, t_new = makeRt(T_new)
    scan_new = (R_new @ scan.T).T + t_new

    app = q3d.QApplication([])

    # create viewer
    viewer = q3d.Viewer(name='example')
    # add all items to viewer
    viewer.add_items({
        'grid': q3d.GridItem(size=10, spacing=1),
        'map': q3d.CloudItem(size=0.01, alpha=0.5, point_type='SPHERE', \
                              color_mode='FLAT', color='#00ff00', depth_test=True),
        'scan': q3d.CloudItem(size=0.01, alpha=0.5, point_type='SPHERE', \
                              color_mode='FLAT', color='#ff0000', depth_test=True),
        'norm': q3d.LineItem(width=2, color='#00ff00', line_type='LINES')})

    viewer['map'].set_data(map)
    viewer['scan'].set_data(scan_new)

    viewer.show()
    app.exec()

