#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import time
from point_cloud_registration import expSO3
import os
import urllib.request
try:
    from q3dviewer.utils.cloud_io import load_pcd
except ImportError:
    print("To visualize the results, please install q3dviewer first by using 'pip install q3dviewer==1.1.6'")
    exit(1)

# get this file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "..", "data")
test_file = os.path.join(data_dir, "B-01.pcd")

def generate_test_data(so3=np.zeros(3), t=np.array([0, 0, 0.3]), num_points=100000):
    # Generate synthetic data for testing
    try:
        map = load_pcd(test_file)
    except FileNotFoundError:
        url = "https://github.com/scomup/point-cloud-registration/raw/main/data/B-01.pcd"
        print(f"File not found. Downloading from {url}...")
        urllib.request.urlretrieve(url, test_file)
        print(f"File downloaded and saved to {test_file}. Please move it to the 'data' directory if needed.")
        map = load_pcd(test_file)
    # map, _ = q3d.load_pcd("/home/liu/.ros/lidar_camera_calib/clouds/0.pcd")
    map = map['xyz']
    R = expSO3(np.array(so3))
    scan = (R @ map.T).T + t
    # use numpy to randomly sample points
    if num_points > scan.shape[0]:
        num_points = scan.shape[0]
    # randomly sample points
    indices = np.random.choice(scan.shape[0], num_points, replace=False)
    scan = scan[indices]
    # add noise
    noise = np.random.normal(0, 0.005, scan.shape)
    scan += noise
    return map, scan
