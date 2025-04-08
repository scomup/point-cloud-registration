"""
Copyright 2025 Liu Yang
Distributed under MIT license. See LICENSE for more information.
"""

import numpy as np
# from collections import defaultdict
from point_cloud_registration.kdtree import KDTree
import time


def get_keys(points, voxel_size=1.0):
    """
    a faster hash for 3d points
    """
    voxel_indices = (np.floor(points / voxel_size)).astype(np.int64)
    HASH_P = 116101
    MAX_N = 10000000000
    x, y, z = voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]
    keys = ((((z * HASH_P) % MAX_N + y) * HASH_P) % MAX_N + x)
    return keys

def fast_unique(keys):
    # Step 1: Sort the keys and get the sorting indices
    # t1 = time.time()
    sorted_indices = np.argsort(keys)
    # t2 = time.time()
    sorted_keys = keys[sorted_indices]
    
    # Step 2: Find where adjacent sorted keys differ (marking unique groups)
    diff = np.concatenate(([True], sorted_keys[1:] != sorted_keys[:-1]))
    # t3 = time.time()
    
    # Step 3: Assign unique IDs to each group (cumulative sum of diff)
    unique_ids = np.cumsum(diff) - 1  # Zero-based indexing
    #  #t4 = time.time()
    
    # Step 4: Reconstruct the inverse mapping for the original array
    inverse_indices = np.empty_like(keys)
    inverse_indices[sorted_indices] = unique_ids
    t5 = time.time()
    # Print time taken for each step
    # print(f"Step 1 (Sorting keys): {(t2 - t1) * 1000:.2f} ms")
    # print(f"Step 2 (Finding differences): {(t3 - t2) * 1000:.2f} ms")
    # print(f"Step 3 (Assigning unique IDs): {(t4 - t3) * 1000:.2f} ms")
    # print(f"Step 4 (Reconstructing inverse mapping): {(t5 - t4) * 1000:.2f} ms")
    # print(f"  fast_unique time: {(t5 - t1) * 1000:.2f} ms")
    
    return inverse_indices

class VoxelGrid:
    """
    An efficient VoxelGrid structure using hash table
    """

    def __init__(self, voxel_size, min_points=10):
        self.voxel_size = voxel_size
        self.kdtree = None
        self.min_points = min_points

    def calc_sqrt_icov(self):
        """
        Calculate the square root of the inverse covariance matrix
        """
        L_lower = np.linalg.cholesky(self.icov)
        L = np.transpose(L_lower, axes=(0, 2, 1))
        self.sqrt_icov = L

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
        # t1 = time.time()
        # Compute voxel keys
        keys = get_keys(points, self.voxel_size)
        # t2 = time.time()
        _, indices = np.unique(keys, return_inverse=True)
        counts = np.bincount(indices)
        # t4 = time.time()

        # Compute centroids of each voxel
        summed_x = np.bincount(indices, weights=points[:, 0])
        summed_y = np.bincount(indices, weights=points[:, 1])
        summed_z = np.bincount(indices, weights=points[:, 2])

        mean_x = summed_x / counts
        mean_y = summed_y / counts
        mean_z = summed_z / counts
        means = np.vstack((mean_x, mean_y, mean_z)).T
        # t5 = time.time()

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
        counts_1 = np.maximum(counts - 1, 1)
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
        
        # Filter out voxels with too few points
        mask = counts >= self.min_points
        means = means[mask]
        covs = covs[mask]
        # t6 = time.time()

        # get the normal of each voxel
        _, eigenvectors = np.linalg.eigh(covs)
        norms = eigenvectors[:, :, 0]
        # t7 = time.time()

        # Create data for voxels
        self.norm = norms
        self.cov = covs
        self.mean = means
        self.kdtree = KDTree(means)
        # print(f"get keys time: {(t2 - t1) * 1000:.2f} ms")
        # print(f"get unique time: {(t4 - t2) * 1000:.2f} ms")
        # print(f"compute mean time: {(t5 - t4) * 1000:.2f} ms")
        # print(f"compute cov time: {(t6 - t5) * 1000:.2f} ms")

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
    Faster coloring of point cloud by voxel using hash table
    """
    keys = get_keys(points, voxel_size)
    unique_ids, inverse_indices = np.unique(keys, return_inverse=True)

    # Generate colors per unique voxel
    np.random.seed(42)
    colors = np.random.randint(0, 256, size=(len(unique_ids), 3), dtype=np.uint8)

    point_colors = colors[inverse_indices]

    # Convert to packed RGB
    rgb = (
        point_colors[:, 0].astype(np.uint32) << 16 |
        point_colors[:, 1].astype(np.uint32) << 8 |
        point_colors[:, 2].astype(np.uint32)
    )

    # Pack into structured array
    data_type = [('xyz', '<f4', (3,)), ('irgb', '<u4')]
    result = np.rec.fromarrays([points.astype(np.float32), rgb], dtype=data_type)
    return result


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
