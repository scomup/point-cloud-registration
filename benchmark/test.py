#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import time
from point_cloud_registration import ICP, PlaneICP, NDT, VPlaneICP, voxel_filter, estimate_normals, expSO3

import q3dviewer as q3d
from test_data import generate_test_data



if __name__ == '__main__':
    map_points, scan_points = generate_test_data()

    max_iter = 30  # Maximum iterations for ICP and NDT
    tol = 1e-3  # Tolerance for convergence
    voxel_size = 1  # Voxel size
    max_dist = voxel_size * 2  # Maximum correspondence distance
    k = 5  # Number of nearest neighbors for normal estimation

    t1 = time.time()
    t2 = time.time()
    icp = NDT(max_iter=max_iter, tol=tol, max_dist=max_dist)
    icp.set_target(map_points)
    t3 = time.time()
    T_new = icp.align(scan_points, init_T=np.eye(4), verbose=True)
    t4 = time.time()

    print("Set target time:", t3 - t2)
    print("PlaneICP time:", t4 - t3)

    print("result", T_new[:3, 3 ])


