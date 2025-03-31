import numpy as np
import time
from point_cloud_registration import ICP
from point_cloud_registration import Point2PlaneICP
from point_cloud_registration import NDT
from point_cloud_registration import VoxelizedPoint2PlaneICP
from point_cloud_registration import estimate_normals
from point_cloud_registration import expSO3
import open3d as o3d
import q3dviewer as q3d


def generate_test_data():
    # Generate synthetic data for testing
    map, _ = q3d.load_pcd("/home/liu/tmp/recorded_frames/clouds/0.pcd")
    map = map['xyz']
    R = expSO3(np.array([0.0, 0.0, 0.0]))
    t = np.array([0.3, 0.3, 0.3])
    scan = (R @ map.T).T + t
    return map, scan


def test_icp(map_points, scan_points, max_iter, tol, max_dist):
    start_time = time.time()
    icp = ICP(max_iter=max_iter, tol=tol, max_dist=max_dist)
    icp.set_target(map_points)
    T_new = icp.align(scan_points, init_T=np.eye(4))
    elapsed_time = time.time() - start_time
    return T_new, elapsed_time


def test_ppicp(map_points, scan_points, max_iter, tol, max_dist, voxel_size):
    start_time = time.time()
    icp = Point2PlaneICP(
        voxel_size=voxel_size, max_iter=max_iter, max_dist=max_dist, tol=tol)
    icp.set_target(map_points)
    T_new = icp.align(scan_points, init_T=np.eye(4))
    elapsed_time = time.time() - start_time
    return T_new, elapsed_time


def test_ndt(map_points, scan_points, max_iter, tol, max_dist, voxel_size):
    start_time = time.time()
    ndt = NDT(
        voxel_size=voxel_size, max_iter=max_iter, max_dist=max_dist, tol=tol)
    ndt.set_target(map_points)
    T_new = ndt.align(scan_points, init_T=np.eye(4))
    elapsed_time = time.time() - start_time
    return T_new, elapsed_time


def test_vppicp(map_points, scan_points, max_iter, tol, max_dist, voxel_size):
    start_time = time.time()
    icp = VoxelizedPoint2PlaneICP(
        voxel_size=voxel_size, max_iter=max_iter, max_dist=max_dist, tol=tol)
    icp.set_target(map_points)
    T_new = icp.align(scan_points, init_T=np.eye(4))
    elapsed_time = time.time() - start_time
    return T_new, elapsed_time


def test_open3d_icp(map_points, scan_points, max_iter, tol, max_dist):
    start_time = time.time()
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(scan_points)
    target.points = o3d.utility.Vector3dVector(map_points)
    result = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance=max_dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        init=np.eye(4),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter, relative_fitness=tol, relative_rmse=tol))
    elapsed_time = time.time() - start_time
    return result.transformation, elapsed_time


def test_open3d_ppicp(map_points, scan_points, max_iter, tol, max_dist, k):
    start_time = time.time()
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(scan_points)
    target.points = o3d.utility.Vector3dVector(map_points)
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    result = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance=max_dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        init=np.eye(4),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter, relative_fitness=tol, relative_rmse=tol))
    elapsed_time = time.time() - start_time
    return result.transformation, elapsed_time



if __name__ == '__main__':
    map_points, scan_points = generate_test_data()

    # Set parameters
    max_iter = 30
    tol = 1e-3
    max_dist = 2
    voxel_size = 0.5
    k = 15

    # Test algorithms
    print("icp...")
    _, time_icp = test_icp(map_points, scan_points, max_iter, tol, max_dist)
    print("ppicp...")
    _, time_ppicp = test_ppicp(map_points, scan_points, max_iter, tol, max_dist, voxel_size)
    print("ndt...")
    _, time_ndt = test_ndt(map_points, scan_points, max_iter, tol, max_dist, voxel_size)
    print("vppicp...")
    _, time_vppicp = test_vppicp(map_points, scan_points, max_iter, tol, max_dist, voxel_size)
    print("open3d_icp...")
    _, time_open3d_icp = test_open3d_icp(map_points, scan_points, max_iter, tol, max_dist)
    print("open3d_ppicp...")
    _, time_open3d_ppicp = test_open3d_ppicp(map_points, scan_points, max_iter, tol, max_dist, k)

    # Output comparison table
    print("\nSpeed Comparison Table:")
    print(f"{'Algorithm':<25}{'Execution Time (s)':>20}")
    print("-" * 45)
    print(f"{'Our ICP':<25}{time_icp:>20.6f}")
    print(f"{'Our Point-to-Plane ICP':<25}{time_ppicp:>20.6f}")
    print(f"{'Our NDT':<25}{time_ndt:>20.6f}")
    print(f"{'Open3D ICP':<25}{time_open3d_icp:>20.6f}")
    print(f"{'Open3D Point-to-Plane ICP':<25}{time_open3d_ppicp:>20.6f}")
