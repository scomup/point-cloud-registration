import numpy as np
import time
from point_cloud_registration.icp import ICP
from point_cloud_registration.voxel import voxel_filter, voxel_filter_old
import open3d as o3d
from point_cloud_registration.math_tools import expSO3
import q3dviewer as q3d


def generate_test_data():
    # Generate synthetic data for testing
    # Generate N x 3 points
    map, _ = q3d.load_pcd("/home/liu/tmp/recorded_frames/clouds/0.pcd")
    return map['xyz']


def test_voxel_filter(voxel_size, points):
    start_time = time.time()
    filtered_points = voxel_filter(points, voxel_size)
    elapsed_time = time.time() - start_time
    return elapsed_time, filtered_points


def test_voxel_filter_old(voxel_size, points):
    start_time = time.time()
    filtered_points = voxel_filter_old(points, voxel_size)
    elapsed_time = time.time() - start_time
    return elapsed_time, filtered_points


def test_open3d_voxel_filter(voxel_size, points):
    start_time = time.time()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    elapsed_time = time.time() - start_time
    filtered_points = np.asarray(downsampled_pcd.points)
    return elapsed_time, filtered_points

if __name__ == '__main__':
    map_points = generate_test_data()

    voxel_size = 0.25  # Adjust voxel size as needed

    t1, our_filtered = test_voxel_filter(voxel_size, map_points)

    t2, our_filtered_old = test_voxel_filter_old(voxel_size, map_points)

    t3, open3d_filtered = test_open3d_voxel_filter(voxel_size, map_points)

    app = q3d.QApplication([])
    # create viewer
    viewer = q3d.Viewer(name='example')
    # add all items to viewer
    viewer.add_items({
        'grid': q3d.GridItem(size=10, spacing=1),
        'scan': q3d.CloudItem(size=0.1, alpha=1, point_type='SPHERE', color_mode='FLAT', color='#00ff00', depth_test=True),
        'scan2': q3d.CloudItem(size=0.1, alpha=1, point_type='SPHERE', color_mode='FLAT', color='#ff0000', depth_test=True),
        'norm': q3d.LineItem(width=2, color='#00ff00', line_type='LINES')})
    viewer['scan'].set_data(our_filtered)
    viewer['scan2'].set_data(open3d_filtered)

    print("\nComparison:")
    print(f"Open3D voxel_filter time: {t3:.6f} seconds")
    print(f"Our voxel_filter_old time: {t2:.6f} seconds")
    print(f"Our voxel_filter_optimized time: {t1:.6f} seconds")
    viewer.show()
    app.exec()
