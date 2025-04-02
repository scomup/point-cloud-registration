import numpy as np
from collections import defaultdict
from point_cloud_registration.kdtree import KDTree
from operator import itemgetter, attrgetter
from concurrent.futures import ThreadPoolExecutor, as_completed


def svd_sqrt(A):
    U, S, Vt = np.linalg.svd(A)  # SVD decomposition
    S_sqrt = np.sqrt(S)          # Square root of singular values
    B = np.diag(S_sqrt) @ Vt     # Compute B
    return B


def get_keys(points, voxel_size=1):
    """
    a faster hash for 3d points
    """
    # voxel_indices = (points // voxel_size).astype(np.int64)
    voxel_indices = (points // voxel_size).astype(np.int64)
    P1 = 2654435761  # Large prime (from Knuth)
    P2 = 5915587277  # Another large prime
    # Fast hash computation using multiply-shift-xor
    keys = (voxel_indices[:, 0] * P1) ^ (voxel_indices[:, 1]
                                         * P2) ^ voxel_indices[:, 2]
    return keys


class VoxelGrid:
    """
    An efficient VoxelGrid structure using hash table
    """

    def __init__(self, voxel_size, min_points=6):
        self.voxel_size = voxel_size
        self.kdtree = None
        self.min_points = min_points

    def add_points(self, points):
        points = points.astype(np.float32)

        # Compute voxel keys
        keys = get_keys(points, self.voxel_size)
        _, inverse_indices = np.unique(keys, return_inverse=True)

        # Count points per voxel
        counts = np.bincount(inverse_indices)
        mask = counts >= self.min_points

        # Filter valid points
        filter = mask[inverse_indices]
        points = points[filter]
        keys = keys[filter]

        _, indices = np.unique(keys, return_inverse=True)
        counts = np.bincount(indices)

        # Compute centroids
        summed_x = np.bincount(indices, weights=points[:, 0])
        summed_y = np.bincount(indices, weights=points[:, 1])
        summed_z = np.bincount(indices, weights=points[:, 2])

        mean_x = summed_x / counts
        mean_y = summed_y / counts
        mean_z = summed_z / counts
        means = np.vstack((mean_x, mean_y, mean_z)).T

        # Compute deviations
        dev = points - means[indices]

        # Compute element-wise covariance terms
        dev_x, dev_y, dev_z = dev[:, 0], dev[:, 1], dev[:, 2]
        cov_xx = dev_x * dev_x
        cov_yy = dev_y * dev_y
        cov_zz = dev_z * dev_z
        cov_xy = dev_x * dev_y
        cov_xz = dev_x * dev_z
        cov_yz = dev_y * dev_z

        # Compute covariance matrices
        valid_counts = np.maximum(counts - 1, 1)  # Prevent division by zero
        covar00 = np.bincount(indices, weights=cov_xx) / valid_counts
        covar01 = np.bincount(indices, weights=cov_xy) / valid_counts
        covar02 = np.bincount(indices, weights=cov_xz) / valid_counts
        covar11 = np.bincount(indices, weights=cov_yy) / valid_counts
        covar12 = np.bincount(indices, weights=cov_yz) / valid_counts
        covar22 = np.bincount(indices, weights=cov_zz) / valid_counts
        # Stack covariance matrices
        covs = np.stack(
            [[covar00, covar01, covar02],
             [covar01, covar11, covar12],
             [covar02, covar12, covar22]]).transpose(2, 0, 1)

        # get the normal of each voxel
        _, eigenvectors = np.linalg.eigh(covs)
        norms = eigenvectors[:, :, 0]

        # Create data for voxels
        self.norms = norms
        self.covs = covs
        self.means = means
        self.kdtree = KDTree(means)

    def find(self, point):
        # Use kdtree to find the nearest cell
        _, idx = self.kdtree.query(point)
        key = list(self.voxels.keys())[idx]
        return self.voxels[key]

    def query(self, points, names):
        """
        Query multiple voxels at once.
        """
        # Use kdtree to find the nearest cell
        _, idx = self.kdtree.query(points)
        data = []
        for n in names:
            if n == 'mean':
                data.append(self.means[idx])
            elif n == 'norm':
                data.append(self.norms[idx])
            elif n == 'cov':
                data.append(self.covs[idx]) 
        return data


def color_by_voxel(points, voxel_size):
    """
    given a set of points, color them based on the voxel they belong to
    """
    keys = get_keys(points, voxel_size)
    # Create random colors for unique keys
    unique_keys = np.unique(keys)
    np.random.seed(42)  # Set seed for reproducibility
    colors = {key: np.random.randint(0, 256, size=3) for key in unique_keys}

    # Assign colors to points based on their keys
    point_colors = np.array([colors[key] for key in keys])
    rgb = point_colors[:, 0] << 24 | point_colors[:,
                                                  1] << 16 | point_colors[:, 2] << 8
    data_type = [('xyz', '<f4', (3,)), ('irgb', '<u4')]
    point_colors = np.rec.fromarrays(
        [points, rgb], dtype=data_type)
    return point_colors


def voxel_filter_old(points, voxel_size):
    """
    original voxel filter for point clouds
    please do not use this function
    """

    keys = get_keys(points, voxel_size)
    # The points in the same voxel will have the same key
    _, unique_indices = np.unique(keys, return_inverse=True)

    # Sort by unique_indices, points in same voxel are grouped together
    idx = np.argsort(unique_indices)
    sorted_points = points[idx]
    sorted_unique_indices = unique_indices[idx]

    # Find the start and end indices of points in each voxel using prefix sum
    prefix_sum = np.cumsum(np.r_[0, np.diff(sorted_unique_indices) != 0])
    ranges = np.where(prefix_sum[:-1] != prefix_sum[1:])[0] + 1
    ranges = np.r_[0, ranges, len(sorted_points)]  # Add first & last indices
    # Add points to each voxel
    filtered_points = []
    for i in range(len(ranges) - 1):
        start, end = ranges[i], ranges[i + 1]
        group_points = sorted_points[start:end]
        filtered_points.append(group_points.mean(axis=0))
    return np.array(filtered_points, dtype=np.float32)


def voxel_filter(points, voxel_size):
    """
    Fast voxel filter for point clouds using hash table

    Parameters:
    - points (np.ndarray): Input point cloud of shape (N, 3).
    - voxel_size (float): Size of each voxel grid.

    Returns:
    - np.ndarray: Filtered point cloud of shape (M, 3), where M <= N.
    """
    import time
    t1 = time.time()
    keys = get_keys(points, voxel_size)
    t2 = time.time()
    _, inverse_indices = np.unique(keys, return_inverse=True)
    t3 = time.time()
    summed_x = np.bincount(inverse_indices, weights=points[:, 0])
    summed_y = np.bincount(inverse_indices, weights=points[:, 1])
    summed_z = np.bincount(inverse_indices, weights=points[:, 2])
    counts = np.bincount(inverse_indices).astype(np.float32)[:, np.newaxis]
    counts[counts == 0] = 1
    filtered_x = summed_x / counts[:, 0]
    filtered_y = summed_y / counts[:, 0]
    filtered_z = summed_z / counts[:, 0]
    filtered_points = np.stack(
        (filtered_x, filtered_y, filtered_z), axis=1).astype(np.float32)
    t4 = time.time()
    # print(f"Time taken for key generation: {t2 - t1:.6f} seconds")
    # print(f"Time taken for unique indices: {t3 - t2:.6f} seconds")
    # print(f"Time taken for voxel filtering: {t4 - t3:.6f} seconds")
    # print(f"Total time taken: {t4 - t1:.6f} seconds")
    return filtered_points
