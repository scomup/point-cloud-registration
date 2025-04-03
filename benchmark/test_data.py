#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import time
from point_cloud_registration import expSO3
import q3dviewer as q3d


def generate_test_data():
    # Generate synthetic data for testing
    map, _ = q3d.load_pcd("data/B-01.pcd")
    map = map['xyz']
    R = expSO3(np.array([0.0, 0.0, 0.0]))
    t = np.array([0.0, 0.0, 0.3])
    scan = (R @ map.T).T + t
    # use numpy to randomly sample points
    num_points = 100000
    indices = np.random.choice(scan.shape[0], num_points, replace=False)
    scan = scan[indices]
    # add noise
    noise = np.random.normal(0, 0.005, scan.shape)
    scan += noise
    return map, scan
