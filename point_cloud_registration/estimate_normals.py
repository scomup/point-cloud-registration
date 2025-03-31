"""
Copyright 2025 Liu Yang
Distributed under MIT license. See LICENSE for more information.
"""

import numpy as np
from point_cloud_registration.kdtree import KDTree


def estimate_normals(points, k=30, get_cov=False):
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
    _, indices = kdtree.query(points, k=k)
    # Gather neighbor points: shape (N, k, 3)
    neighbors = points[indices]
    neighbors = neighbors.astype(np.float32)

    # Compute the covariance matrix for each point's neighbors
    mean = neighbors.mean(axis=1, keepdims=True)
    centered = neighbors - mean
    covs = np.einsum('nki,nkj->nij', centered, centered) / (k - 1)
    _, eigvecs = np.linalg.eigh(covs) 
    # get the normal from the smallest eigenvector
    normals = eigvecs[:, :, 0]  # shape: (N, 3)
    data_type = [('xyz', '<f4', (3,)), ('norm', '<f4', (3,))]
    if get_cov:
        # Return covariance matrices along with normals
        return normals, covs
    else:
        # Return only normals
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