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

    def calc_icov(self):
        """
        fast inverse of covariance matrix
        20x faster than np.linalg.inv(self.cov)
        """
        a, b, c = self.cov[:, 0, 0], self.cov[:, 1, 1], self.cov[:, 2, 2]
        d, e, f = self.cov[:, 0, 1], self.cov[:, 0, 2], self.cov[:, 1, 2]
        f2, d2, e2 = f * f, d * d, e * e
        bc, ac, ab = b * c, a * c, a * b
        dc, de, ef = d * c, d * e, e * f
        af, df, eb = a * f, d * f, e * b
        det_A = a * bc + 2 * de * f - a * f2 - b * e2 - c * d2

        # if np.any(det_A == 0):
        #     raise ValueError("Singular covariance matrix detected. \
        #              The number of points may be too small. \
        #              Use a larger voxel size.")

        det_A[det_A == 0] = 1000000 # set a large value to avoid singular matrix

        # Compute cofactor matrix
        c0 =  (bc - f2) / det_A
        c1 = -(dc - ef) / det_A
        c2 =  (df - eb) / det_A
        c3 =  (ac - e2) / det_A
        c4 = -(af - de) / det_A
        c5 = ( ab - d2) / det_A
        icov = np.array([
            [c0, c1, c2],
            [c1, c3, c4],
            [c2, c4, c5]])

        # Compute inverse using adjugate and determinant
        self.icov = icov.transpose(2, 0, 1)  # shape: (N, 3, 3)

    def set_points(self, points):
        points = points.astype(np.float32)

        # Compute voxel keys
        keys = get_keys(points, self.voxel_size)
        _, indices0 = np.unique(keys, return_inverse=True)

        # Count points per voxel
        counts = np.bincount(indices0)
        mask = counts >= self.min_points

        # Filter valid voxels which have enough points
        filter = mask[indices0]
        points = points[filter]
        keys = keys[filter]

        # recompute the indices after filtering
        _, indices = np.unique(keys, return_inverse=True)
        counts = np.bincount(indices)

        # Compute centroids of each voxel
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

        # Compute covariance of each voxel
        counts_1 = counts - 1  # the min of counts shloud over self.min_points
        c00 = np.bincount(indices, weights=cov_xx) / counts_1
        c01 = np.bincount(indices, weights=cov_xy) / counts_1
        c02 = np.bincount(indices, weights=cov_xz) / counts_1
        c11 = np.bincount(indices, weights=cov_yy) / counts_1
        c12 = np.bincount(indices, weights=cov_yz) / counts_1
        c22 = np.bincount(indices, weights=cov_zz) / counts_1
        # Stack covariance matrices
        covs = np.stack(
            [[c00, c01, c02],
             [c01, c11, c12],
             [c02, c12, c22]]).transpose(2, 0, 1)

        # get the normal of each voxel
        _, eigenvectors = np.linalg.eigh(covs)
        norms = eigenvectors[:, :, 0]

        # Create data for voxels
        self.norm = norms
        self.cov = covs
        self.mean = means
        self.kdtree = KDTree(means)

    def query(self, points, names):
        """
        Query multiple voxels at once.
        """
        # Use kdtree to find the nearest cell
        dist, idx = self.kdtree.query(points)
        query_data = {name: getattr(self, name)[idx] for name in names}
        query_data['dist'] = dist
        return query_data


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
    #t1 = time.time()
    keys = get_keys(points, voxel_size)
    #t2 = time.time()
    _, inverse_indices = np.unique(keys, return_inverse=True)
    #t3 = time.time()
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
    # t4 = time.time()
    # print(f"get keys time: {(t2 - t1) :.2f} sec")
    # print(f"get unique time: {(t3 - t2) :.2f} sec")
    # print(f"compute mean time: {(t4 - t3) :.2f} sec")
    # print(f"total time: {(t4 - t1) :.2f} sec")
    return filtered_points
