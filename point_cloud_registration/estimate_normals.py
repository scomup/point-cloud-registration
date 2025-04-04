"""
Copyright 2025 Liu Yang
Distributed under MIT license. See LICENSE for more information.
"""

import numpy as np
from point_cloud_registration.kdtree import KDTree
import time


def estimate_normals(points, k=15):
    """
    Estimate the normals of a point cloud using k-nearest neighbors.
    Args:
        points (numpy.ndarray): The input point cloud of shape (N, 3).
        k (int): The number of nearest neighbors to consider.
    Returns:
        numpy.ndarray: The estimated normals of shape (N, 3).
    """
    # Create a KDTree for efficient nearest neighbor search
    kdtree = KDTree(points)
    # Query k nearest neighbors for each point
    normals = estimate_norm_with_tree(points, kdtree, k=k)
    return normals
    

def estimate_norm_with_tree(points, kdtree, k=15):
    """
    Estimate the normals of a point cloud using k-nearest neighbors.
    Args:
        points (numpy.ndarray): The input point cloud of shape (N, 3).
        k (int): The number of nearest neighbors to consider.
    Returns:
        numpy.ndarray: The estimated normals of shape (N, 3).
    """
    # Query k nearest neighbors for each point

    # t1 = time.time()
    _, indices = kdtree.query(points, k=k)
    # t2 = time.time()
    n = len(points)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    # t3 = time.time()

    sum = np.zeros((n, 3), dtype=np.float32)
    ppt = np.zeros((3, 3, n), dtype=np.float32)

    # the k is not very large, so we can use the for loop
    for i in range(k):
        idx = indices[:, i]
        sum += points[idx]
        ppt[0, 0] += xx[idx]
        ppt[1, 1] += yy[idx]
        ppt[2, 2] += zz[idx]
        ppt[0, 1] += xy[idx]
        ppt[0, 2] += xz[idx]
        ppt[1, 2] += yz[idx]
    # Copy upper triangular to lower triangular
    ppt[1, 0] = ppt[0, 1]
    ppt[2, 0] = ppt[0, 2]
    ppt[2, 1] = ppt[1, 2]
    # t4 = time.time()
    # Compute the mean and covariance
    ppt = ppt.transpose(2, 0, 1)
    means = sum / k
    covs = ppt / k - np.einsum('ij,ik->ijk', means, means)
    # t5 = time.time()

    # Compute normals as the eigenvector corresponding to the smallest eigenvalue
    _, eigvecs = np.linalg.eigh(covs)
    normals = eigvecs[:, :, 0]
    # t6 = time.time()
    # print(f"num points: {n}")
    # print(f"KDTree query time: {(t2 - t1) * 1000:.2f} ms")
    # print(f"Compute cross time: {(t3 - t2) * 1000:.2f} ms")
    # print(f"sum ppt time: {(t4 - t3) * 1000:.2f} ms")
    # print(f"compute mean cov time: {(t5 - t4) * 1000:.2f} ms")
    # print(f"compute normal time: {(t6 - t5) * 1000:.2f} ms")
    # Filter out points with too few neighbors

    return normals



def get_norm_lines(points, normals, length=0.1):
    """
    Generate lines for visualization of normals.
    Args:
        points (numpy.ndarray): The input point cloud of shape (N, 3).
        normals (numpy.ndarray): The estimated normals of shape (N, 3).
        length (float): The length of the normal lines.
    Returns:
        numpy.ndarray: The lines for visualization of shape (2N, 3).
    """
    offset_points = points + normals * length
    lines = np.empty((2 * points.shape[0], points.shape[1]), dtype=points.dtype)
    lines[::2] = points
    lines[1::2] = offset_points
    return lines


if __name__ == "__main__":
    import q3dviewer as q3d
    data, _ = q3d.load_pcd("/home/liu/tmp/recorded_frames/clouds/0.pcd")
    estimate_normals(data['xyz'])