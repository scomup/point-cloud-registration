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
    normals, _, _ = estimate_norm_cov_mean_with_tree(points, kdtree, k=k)
    return normals
    

def estimate_norm_cov_mean_with_tree(points, kdtree, k=15):
    """
    Estimate the normals of a point cloud using k-nearest neighbors.
    Args:
        points (numpy.ndarray): The input point cloud of shape (N, 3).
        k (int): The number of nearest neighbors to consider.
    Returns:
        numpy.ndarray: The estimated normals of shape (N, 3).
    """
    # Query k nearest neighbors for each point
    _, indices = kdtree.query(points, k=k)
    # Gather neighbor points: shape (N, k, 3)
    neighbors = points[indices]
    neighbors = neighbors.astype(np.float32)

    # Compute the covariance matrix for each point's neighbors
    means = neighbors.mean(axis=1)
    centered = neighbors - means[:, np.newaxis, :]

    """
    # original covariance matrix calculation version
    """
    # covs = np.einsum('nki,nkj->nij', centered, centered) / (k - 1)
    # _, eigvecs = np.linalg.eigh(covs) 

    """
    # optimized covariance matrix calculation
    # reuse the upper triangular part of the covariance matrix
    # covariance matrix is symmetric, so we can compute only half of it
    # this way will much faster than the original version
    """
    x, y, z = centered[:, :, 0], centered[:, :, 1], centered[:, :, 2]
    # Compute elements of the covariance matrix
    xx = np.einsum('ni,ni->n', x, x) / (k - 1)
    yy = np.einsum('ni,ni->n', y, y) / (k - 1)
    zz = np.einsum('ni,ni->n', z, z) / (k - 1)
    xy = np.einsum('ni,ni->n', x, y) / (k - 1)
    xz = np.einsum('ni,ni->n', x, z) / (k - 1)
    yz = np.einsum('ni,ni->n', y, z) / (k - 1)
    
    # conbine the covariance matrix elements into a 3x3 matrix
    covs = np.array([[xx, xy, xz],
                     [xy, yy, yz],
                     [xz, yz, zz]]).transpose(2, 0, 1)

    _, eigvecs = np.linalg.eigh(covs)

    # get the normal from the smallest eigenvector
    normals = eigvecs[:, :, 0]  # shape: (N, 3)
    return normals, covs, means



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