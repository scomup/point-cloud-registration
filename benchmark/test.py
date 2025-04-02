#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import time
from point_cloud_registration import ICP, PlaneICP, NDT, VPlaneICP, voxel_filter, estimate_normals, expSO3
import q3dviewer as q3d


def generate_test_data():
    # Generate synthetic data for testing
    map, _ = q3d.load_pcd("/home/liu/tmp/recorded_frames/clouds/0.pcd")
    map = map['xyz']
    R = expSO3(np.array([0.0, 0.0, 0.0]))
    t = np.array([0.1, 0.0, 0.0])
    scan = (R @ map.T).T + t
    return map, scan


if __name__ == '__main__':
    map_points, scan_points = generate_test_data()

    # Parameters (consistent with C++ implementation)
    max_iter = 30  # Maximum iterations for ICP and NDT
    tol = 1e-3  # Tolerance for convergence
    max_dist = 0.5  # Maximum correspondence distance
    voxel_size = 0.5  # Voxel size
    k = 15  # Number of nearest neighbors for normal estimation
    downsampling_resolution = 0.1

    map_points = voxel_filter(map_points, downsampling_resolution)
    scan_points = voxel_filter(scan_points, downsampling_resolution)
    t1 = time.time()
    icp = VPlaneICP(max_iter=max_iter, tol=tol, max_dist=max_dist)
    icp.set_target(map_points)
    t2 = time.time()
    T_new = icp.align(scan_points, init_T=np.eye(4), verbose=False)
    t3 = time.time()
    print("Set target time:", t2 - t1)
    print("VPlaneICP time:", t3 - t2)
    print("Total time:", t3 - t1)

