import numpy as np
import time
from point_cloud_registration.icp import ICP
from point_cloud_registration.voxelized_point_plane_icp import VoxelizedPoint2PlaneICP
import open3d as o3d
import q3dviewer as q3d
from point_cloud_registration.math_tools import expSO3


def generate_test_data():
    # Generate synthetic data for testing
    # Generate N x 3 points
    map, _ = q3d.load_pcd("/home/liu/tmp/recorded_frames/clouds/0.pcd")
    # scan, _ = q3d.load_pcd("/home/liu/tmp/recorded_frames/clouds/2.pcd")
    map = map['xyz']
    #  scan = scan['xyz']
    R = expSO3(np.array([0.0, 0.0, 0.0]))
    t = np.array([0.3, 0.3, 0.3])
    scan = (R @ map.T).T + t

    return map, scan


def test_vppicp(map_points, scan_points, max_iter, tol, max_dist, voxel_size):
    start_time = time.time()
    icp = VoxelizedPoint2PlaneICP(
        voxel_size=voxel_size, max_iter=max_iter, max_dist=max_dist, tol=tol)
    icp.set_target(map_points)
    T_new = icp.fit(scan_points, init_T=np.eye(4))
    elapsed_time = time.time() - start_time
    return T_new, elapsed_time


if __name__ == '__main__':
    map_points, scan_points = generate_test_data()

    # Set parameters
    max_iter = 50
    tol = 1e-6
    max_dist = 2

    # Test VoxelizedPoint2PlaneICP
    voxel_size = 0.5
    T_vppicp, time_vppicp = test_vppicp(
        map_points, scan_points, max_iter, tol, max_dist, voxel_size)
    print(
        f"VoxelizedPoint2PlaneICP: Time = {time_vppicp:.4f}s, Transformation:\n{T_vppicp}")

