from point_cloud_registration import VoxelPoint2PlaneICP
from point_cloud_registration.math_tools import makeRt
import numpy as np

try:
    import q3dviewer as q3d
except ImportError:
    print("Please install q3dviewer using 'pip install q3dviewer'")
    exit(0)

if __name__ == '__main__':

    # Generate N x 3 points
    map, _ = q3d.load_pcd("/home/liu/tmp/recorded_frames/clouds/0.pcd")
    scan, _ = q3d.load_pcd("/home/liu/tmp/recorded_frames/clouds/2.pcd")
    map = map['xyz']
    scan = scan['xyz']
    # R = expSO3(np.array([0.0, 0.0, 0.0]))
    # t = np.array([0.3, 0.3, 0.3])
    # scan = (R @ map.T).T + t

    icp = VoxelPoint2PlaneICP(voxel_size=0.5, max_iter=100, max_dist=2, tol=1e-3)
    icp.update_target(map)
    T_new = icp.fit(scan, init_T=np.eye(4))
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

