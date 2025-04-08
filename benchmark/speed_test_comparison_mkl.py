#!/usr/bin/env python3

import numpy as np

# Check if MKL is being used by NumPy
blas_info = np.__config__.get_info('blas_opt_info')
if blas_info and 'mkl' in str(blas_info).lower():
    print("MKL is being used by NumPy.")
else:
    print("MKL is NOT being used by NumPy.")
    print("To use MKL, install the MKL version of NumPy.")
    print("You can do this by running: pip install intel-numpy")
    print("After installation, reinstall pykdtree 1.3.7 from source.")
    exit(1)


import time
from point_cloud_registration import ICP, PlaneICP, NDT, VPlaneICP, voxel_filter, estimate_normals, KDTree, estimate_norm_with_tree
from test_data import generate_test_data

def test_icp(map_points, scan_points, max_iter, tol, max_dist):
    start_time = time.time()
    icp = ICP(max_iter=max_iter, tol=tol, max_dist=max_dist)
    icp.set_target(map_points)
    T_new = icp.align(scan_points, init_T=np.eye(4))
    elapsed_time = time.time() - start_time
    return T_new, elapsed_time


def test_ppicp(map_points, scan_points, max_iter, tol, max_dist):

    kdtree = KDTree(map_points)
    normal = estimate_norm_with_tree(map_points, kdtree)

    start_time = time.time()
    icp = PlaneICP(max_iter=max_iter,
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


def test_voxel_filter(voxel_size, points):
    start_time = time.time()
    filtered_points = voxel_filter(points, voxel_size)
    elapsed_time = time.time() - start_time
    return elapsed_time, filtered_points


def test_our_estimate_normals(points, k):
    """
    Test the performance of the custom normal estimation.
    """
    start_time = time.time()
    normals = estimate_normals(points, k)
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

    print("\nSpeed Comparison Voxel Filter:")
    print(f"{'Algorithm':<35}{'Execution Time (s)':>20}")
    print("-" * 55)
    print(f"{'Our Voxel Filter':<35}{t1:>20.6f}")

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
    _, time_ppicp = test_ppicp(map_points, scan_points, max_iter, tol, max_dist)
    print(f"{'Our Point-to-Plane ICP':<35}{time_ppicp:>20.6f}")
    _, time_ndt = test_ndt(map_points, scan_points, max_iter, tol, max_dist, voxel_size)
    print(f"{'Our NDT':<35}{time_ndt:>20.6f}")
    _, time_vppicp = test_vppicp(map_points, scan_points, max_iter, tol, max_dist, voxel_size)
    print(f"{'Our Voxelized Point-to-Plane ICP':<35}{time_vppicp:>20.6f}")

    t3, our_normals = test_our_estimate_normals(map_points, k)

    print("\nSpeed Comparison Normal Estimation:")
    print(f"{'Algorithm':<35}{'Execution Time (s)':>20}")
    print("-" * 55)
    print(f"{'Our Normal Estimation':<35}{t3:>20.6f}")
