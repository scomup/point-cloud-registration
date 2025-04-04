from point_cloud_registration import ICP, PlaneICP, NDT, VPlaneICP, voxel_filter, estimate_normals, expSO3

from point_cloud_registration.math_tools import makeRt, expSO3, transform_points, makeT
import numpy as np
from benchmark.test_data import generate_test_data


try:
    import q3dviewer as q3d
except ImportError:
    print("Please install q3dviewer using 'pip install q3dviewer'")
    exit(0)


if __name__ == '__main__':

    # Generate N x 3 points
    map, scan = generate_test_data()
    
    T = np.eye(4)

    icp = PlaneICP(voxel_size=1, max_iter=100, max_dist=100, tol=1e-5)
    icp.set_target(map)
    T_new = icp.align(scan, init_T=T, verbose=True)
    # icp.max_dist = 0.1
    # T_new = icp.align(scan, init_T=T_new, verbose=True)
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
        'map': q3d.CloudItem(size=0.01, alpha=1, point_type='SPHERE', \
                              color_mode='FLAT', color='lime', depth_test=True),
        'scan': q3d.CloudItem(size=0.05, alpha=1, point_type='SPHERE', \
                              color_mode='FLAT', color='r', depth_test=True),
        'norm': q3d.LineItem(width=2, color='lime', line_type='LINES')})

    viewer['map'].set_data(map)
    viewer['scan'].set_data(scan_new)

    viewer.show()
    app.exec()

