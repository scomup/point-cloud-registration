import numpy as np
import time
from point_cloud_registration.icp import ICP
import open3d as o3d
from point_cloud_registration.math_tools import expSO3
import q3dviewer as q3d


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


def test_icp(map_points, scan_points, max_iter, tol, max_dist):
    start_time = time.time()
    icp = ICP(max_iter=max_iter, tol=tol, max_dist=max_dist)
    icp.set_target(map_points)
    T_new = icp.fit(scan_points, init_T=np.eye(4))
    elapsed_time = time.time() - start_time
    return T_new, elapsed_time


def test_open3d_icp(map_points, scan_points, max_iter, tol, max_dist):
    start_time = time.time()
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(scan_points)
    target.points = o3d.utility.Vector3dVector(map_points)
    # Perform normal ICP
    result = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance=max_dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        init=np.eye(4),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter, relative_fitness=tol, relative_rmse=tol))
    elapsed_time = time.time() - start_time
    return result.transformation, elapsed_time


if __name__ == '__main__':
    map_points, scan_points = generate_test_data()

    # Set parameters
    max_iter = 50
    tol = 1e-6
    max_dist = 2

    # Test VoxelPoint2PlaneICP
    T_icp, time_icp = test_icp(
        map_points, scan_points, max_iter, tol, max_dist)
    print(
        f"our ICP: Time = {time_icp:.4f}s, Transformation:\n{T_icp}")


    # Test Open3D ICP
    T_open3d, time_open3d = test_open3d_icp(
        map_points, scan_points, max_iter, tol, max_dist)
    print(
        f"Open3D ICP: Time = {time_open3d:.4f}s, Transformation:\n{T_open3d}")
