import numpy as np
import time
from point_cloud_registration import ICP, PlaneICP, NDT, VPlaneICP, voxel_filter, estimate_normals, KDTree, estimate_norm_with_tree
try:
    import open3d as o3d
except ImportError:
    print("To compare with Open3D, please install Open3D first by using 'pip install open3d")
    exit(1)

from test_data import generate_test_data

def test_icp(map_points, scan_points, max_iter, tol, max_dist):
    start_time = time.time()
    icp = ICP(max_iter=max_iter, tol=tol, max_dist=max_dist)
    icp.set_target(map_points)
    T_new = icp.align(scan_points, init_T=np.eye(4))
    elapsed_time = time.time() - start_time
    return T_new, elapsed_time


def test_ppicp(map_points, scan_points, max_iter, tol, max_dist, voxel_size):

    kdtree = KDTree(map_points)
    normal = estimate_norm_with_tree(map_points, kdtree)

    start_time = time.time()
    icp = PlaneICP(voxel_size=voxel_size,
                   max_iter=max_iter,
                   max_dist=max_dist,
                   tol=tol)
    icp.set_target(map_points, kdtree, normal)
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
    icp = VPlaneICP(
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
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(scan_points)
    target.points = o3d.utility.Vector3dVector(map_points)
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    start_time = time.time()
    result = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance=max_dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        init=np.eye(4),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter, relative_fitness=tol, relative_rmse=tol))
    elapsed_time = time.time() - start_time
    return result.transformation, elapsed_time


def test_voxel_filter(voxel_size, points):
    start_time = time.time()
    filtered_points = voxel_filter(points, voxel_size)
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


def test_our_estimate_normals(points, k):
    """
    Test the performance of the custom normal estimation.
    """
    start_time = time.time()
    normals = estimate_normals(points, k)
    elapsed_time = time.time() - start_time
    return elapsed_time, normals


def test_open3d_estimate_normals(points, k):
    """
    Test the performance of Open3D's normal estimation.
    """
    start_time = time.time()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    normals = np.asarray(pcd.normals)
    elapsed_time = time.time() - start_time
    return elapsed_time, normals


def test_small_gicp(map_points, scan_points, max_iter, tol, max_dist, down_res):
    """
    Test the performance of small_gicp.
    """
    try:
        import small_gicp
    except ImportError:
        print("Please install small_gicp using 'pip install small_gicp'")
        return
    # Ensure input arrays are of type float64
    map_points = map_points.astype(np.float64)
    scan_points = scan_points.astype(np.float64)

    # Supported registration types
    algos = ['ICP', 'PLANE_ICP', 'GICP', 'VGICP']
    for algo in algos:
        start_time = time.time()
        result = small_gicp.align(
            map_points, scan_points,
            registration_type=algo,
            max_correspondence_distance=max_dist,
            downsampling_resolution=0.01, # dont use downsampling
            max_iterations=max_iter,
            verbose=False,
            # translation_epsilon=tol,
            # rotation_epsilon=tol
        )
        elapsed_time = time.time() - start_time
        print(f"{'Small GICP (' + algo + ')':<35}{elapsed_time:>20.6f}")


if __name__ == '__main__':
    map_points, scan_points = generate_test_data()

    # Parameters (consistent with C++ implementation)
    max_iter = 30  # Maximum iterations for ICP and NDT
    tol = 1e-3  # Tolerance for convergence
    voxel_size = 1  # Voxel size
    max_dist = voxel_size * 2  # Maximum correspondence distance
    k = 5  # Number of nearest neighbors for normal estimation

    # Test voxel filters
    print("voxel_filter...")
    t1, our_filtered = test_voxel_filter(voxel_size, map_points)
    print("open3d_voxel_filter...")
    t2, open3d_filtered = test_open3d_voxel_filter(voxel_size, map_points)

    print("\nSpeed Comparison Voxel Filter:")
    print(f"{'Algorithm':<35}{'Execution Time (s)':>20}")
    print("-" * 55)
    print(f"{'Our Voxel Filter':<35}{t1:>20.6f}")
    print(f"{'Open3D Voxel Filter':<35}{t2:>20.6f}")    

    # Test algorithms
    # Output comparison table

    # num_points = len(map_points)
    # print(f"\nNumber of points: {num_points}")
    # print("Note: we do not use the downsampled point cloud for registration!")
    print(f"\nSpeed Comparison Registration Algorithms:")
    print(f"{'Algorithm':<35}{'Execution Time (s)':>20}")
    print("-" * 55)
    _, time_icp = test_icp(map_points, scan_points, max_iter, tol, max_dist)
    print(f"{'Our ICP':<35}{time_icp:>20.6f}")
    _, time_ppicp = test_ppicp(map_points, scan_points, max_iter, tol, max_dist, voxel_size)
    print(f"{'Our Point-to-Plane ICP':<35}{time_ppicp:>20.6f}")
    _, time_ndt = test_ndt(map_points, scan_points, max_iter, tol, max_dist, voxel_size)
    print(f"{'Our NDT':<35}{time_ndt:>20.6f}")
    _, time_vppicp = test_vppicp(map_points, scan_points, max_iter, tol, max_dist, voxel_size)
    print(f"{'Our Voxelized Point-to-Plane ICP':<35}{time_vppicp:>20.6f}")

    _, time_open3d_icp = test_open3d_icp(map_points, scan_points, max_iter, tol, max_dist)
    print(f"{'Open3D ICP':<35}{time_open3d_icp:>20.6f}")
    _, time_open3d_ppicp = test_open3d_ppicp(map_points, scan_points, max_iter, tol, max_dist, k)
    print(f"{'Open3D Point-to-Plane ICP':<35}{time_open3d_ppicp:>20.6f}")
    test_small_gicp(map_points, scan_points, max_iter, tol, max_dist, voxel_size)
    # Test normal estimation
    print("our_estimate_normals...")
    t3, our_normals = test_our_estimate_normals(map_points, k)
    print("open3d_estimate_normals...")
    t4, open3d_normals = test_open3d_estimate_normals(map_points, k)

    print("\nSpeed Comparison Normal Estimation:")
    print(f"{'Algorithm':<35}{'Execution Time (s)':>20}")
    print("-" * 55)
    print(f"{'Our Normal Estimation':<35}{t3:>20.6f}")
    print(f"{'Open3D Normal Estimation':<35}{t4:>20.6f}")
