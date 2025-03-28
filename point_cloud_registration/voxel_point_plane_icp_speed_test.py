import numpy as np
import time
from voxel_point_plane_icp import VoxelPoint2PlaneICP
import open3d as o3d
import q3dviewer as q3d
from math_tools import makeRt, expSO3, makeT


def generate_test_data():
    # Generate synthetic data for testing
        # Generate N x 3 points
    map, _ = q3d.load_pcd("/home/liu/tmp/recorded_frames/clouds/0.pcd")
    scan, _ = q3d.load_pcd("/home/liu/tmp/recorded_frames/clouds/2.pcd")
    map = map['xyz']
    scan = scan['xyz']
    # R = expSO3(np.array([0.0, 0.0, 0.0]))
    # t = np.array([0.3, 0.3, 0.3])
    # scan = (R @ map.T).T + t

    return map, scan

def test_voxel_icp(map_points, scan_points, max_iter, tol):
    icp = VoxelPoint2PlaneICP(voxel_size=0.5, max_iter=max_iter, tol=tol)
    icp.update_target(map_points)
    start_time = time.time()
    T_new = icp.fit(scan_points, init_T=np.eye(4))
    elapsed_time = time.time() - start_time
    return T_new, elapsed_time

def test_open3d_icp(map_points, scan_points, max_iter, tol):
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(scan_points)
    target.points = o3d.utility.Vector3dVector(map_points)

    start_time = time.time()
    # Precompute normals for the target point cloud
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

    result = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance=2.0,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        init=np.eye(4),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter, relative_fitness=tol, relative_rmse=tol))
    elapsed_time = time.time() - start_time
    return result.transformation, elapsed_time

if __name__ == '__main__':
    map_points, scan_points = generate_test_data()

    # Set parameters
    max_iter = 50
    tol = 1e-3

    # Test VoxelPoint2PlaneICP
    T_voxel, time_voxel = test_voxel_icp(map_points, scan_points, max_iter, tol)
    print(f"VoxelPoint2PlaneICP: Time = {time_voxel:.4f}s, Transformation:\n{T_voxel}")

    # Test Open3D ICP
    T_open3d, time_open3d = test_open3d_icp(map_points, scan_points, max_iter, tol)
    print(f"Open3D ICP: Time = {time_open3d:.4f}s, Transformation:\n{T_open3d}")
