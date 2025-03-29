from point_cloud_registration import VoxelizedPoint2PlaneICP
from point_cloud_registration import ICP

from point_cloud_registration.math_tools import makeRt, expSO3, transform_points, makeT
import numpy as np

try:
    import q3dviewer as q3d
except ImportError:
    print("Please install q3dviewer using 'pip install q3dviewer'")
    exit(0)

if __name__ == '__main__':

    # Generate N x 3 points
    map, _ = q3d.load_pcd("/home/liu/tmp/recorded_frames/clouds/0.pcd")

    T = makeT(expSO3(np.array([0.0, 0.0, 0.0])), np.array([0.3, 0.0, 0.0]))
    scan = transform_points(T, map['xyz'])
    # scan, _ = q3d.load_pcd("/home/liu/tmp/recorded_frames/clouds/2.pcd")
    map = map['xyz']

    
    T = np.eye(4)
    # T = np.array([[0.91724685,  0.39554969, -0.04691057, -0.01918991],
    #               [-0.39801572,  0.90555297, -0.14683008, -0.02283431],
    #               [-0.01559759,  0.15334987,  0.98804928, -0.00580279],
    #               [0.,  0.,  0.,  1.]])

    # T = np.eye(4)
    icp = ICP(max_iter=100, max_dist=2, tol=1e-5)
    icp.set_target(map)
    T_new = icp.fit(scan, init_T=T, verbose=True)
    icp.max_dist = 0.1
    T_new = icp.fit(scan, init_T=T_new, verbose=True)
    R_new, t_new = makeRt(T_new)

    scan_new = transform_points(T_new, scan)

    print(T_new)
    # scan_new = (R_new @ scan.T).T + t_new

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

