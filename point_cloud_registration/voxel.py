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


class VoxelCell:
    """
    Store the normal distribution information of a voxel cell,
    used for Normal Distributions Transform and Point-to-Plane ICP.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.num = 0
        self.sum = np.zeros(3)
        self.ppt = np.zeros([3, 3])
        self.mean = np.zeros(3)
        self.icov = np.zeros([3, 3])
        self.sqrt_icov = np.zeros([3, 3])
        self.norm = np.zeros(3)

    def add_points(self, points):
        self.num += points.shape[0]
        self.sum += np.sum(points, axis=0)
        self.ppt += np.dot(points.T, points)
        self.mean = self.sum / self.num
        cov = (self.ppt - 2 * np.outer(self.sum, self.mean)) / \
            self.num + np.outer(self.mean, self.mean)
        _, eigenvectors = np.linalg.eigh(cov)
        self.norm = eigenvectors[:, 0]
        self.icov = np.linalg.inv(cov)
        self.sqrt_icov = svd_sqrt(self.icov) # from ndt
        pass

    def set_points(self, points):
        self.reset()
        self.add_points(points)


class VoxelGrid:
    """
    An efficient VoxelGrid structure using hash table
    """
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size
        self.voxels = defaultdict(VoxelCell)
        self.kdtree = None
        self.voxel_keys = None

    @ staticmethod
    def get_keys(points, voxel_size=1):
        """
        a faster hash for 3d points
        """
        voxel_indices = (points // voxel_size).astype(np.int64)
        P1 = 2654435761  # Large prime (from Knuth)
        P2 = 5915587277  # Another large prime
        # Fast hash computation using multiply-shift-xor
        keys = (voxel_indices[:,0] * P1) ^ (voxel_indices[:,1] * P2) ^ voxel_indices[:,2]
        return keys
    
    def add_points(self, points):
        points = points.astype(np.float32)  # Ensure type is float32
        keys = self.get_keys(points, self.voxel_size)

        # The points in the same voxel will have the same key
        _, unique_indices = np.unique(keys, return_inverse=True)

        # Sort by unique_indices, points in same voxel are grouped together
        idx = np.argsort(unique_indices)
        sorted_points = points[idx]
        sorted_keys = keys[idx]
        sorted_unique_indices = unique_indices[idx]

        # Find the start and end indices of points in each voxel using prefix sum
        prefix_sum = np.cumsum(np.r_[0, np.diff(sorted_unique_indices) != 0])
        ranges = np.where(prefix_sum[:-1] != prefix_sum[1:])[0] + 1
        ranges = np.r_[0, ranges, len(sorted_points)]  # Add first & last indices

        # Add points to each voxel
        for i in range(len(ranges) - 1):
            start, end = ranges[i], ranges[i + 1]
            group_points = sorted_points[start:end]
            if len(group_points) >= 4:  # Only process groups with >= 4 points
                key = sorted_keys[start]
                self.voxels[key].add_points(group_points)

        # Build KDTree using voxel means, for fast nearest neighbor search
        self.voxel_keys, voxel_points = zip(
            *[(key, self.voxels[key].mean) for key in self.voxels])
        self.kdtree = KDTree(np.array(voxel_points, dtype=np.float32))


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
        keys = itemgetter(*idx)(self.voxel_keys)
        # get all the voxel cells
        selected_voxels = itemgetter(*keys)(self.voxels)
        # get the queried data
        data = [[getattr(v, n) for v in selected_voxels] for n in names]
        # data2 = [[v.mean, v.norm] for v in selected_voxels]
        return data


def color_by_voxel(points, voxel_size):
    """
    given a set of points, color them based on the voxel they belong to
    """
    keys = VoxelGrid.get_keys(points, voxel_size)
    # Create random colors for unique keys
    unique_keys = np.unique(keys)
    np.random.seed(42)  # Set seed for reproducibility
    colors = {key: np.random.randint(0, 256, size=3) for key in unique_keys}
    
    # Assign colors to points based on their keys
    point_colors = np.array([colors[key] for key in keys])
    rgb = point_colors[:, 0] << 24 | point_colors[:, 1] << 16 | point_colors[:, 2] << 8
    data_type = [('xyz', '<f4', (3,)), ('irgb', '<u4')]
    point_colors = np.rec.fromarrays(
                    [points, rgb], dtype=data_type)
    return point_colors


if __name__ == '__main__':

    # Generate N x 3 points
    map, _ = q3d.load_pcd("/home/liu/tmp/recorded_frames/clouds/0.pcd")

    T = makeT(expSO3(np.array([0.0, 0.0, 0.0])), np.array([0.3, 0.0, 0.0]))
    scan = transform_points(T, map['xyz'])
    # scan, _ = q3d.load_pcd("/home/liu/tmp/recorded_frames/clouds/2.pcd")
    map = map['xyz']

    
    T = np.eye(4)
    # T = np.array([[0.91724685,  0.39554969, -0.04691057, -0.01918991],
    #               [-0.39801572,  0.90555297, -0.14683008, -0.02283431],
    #               [-0.01559759,  0.15334987,  0.98804928, -0.00580279],
    #               [0.,  0.,  0.,  1.]])

    # T = np.eye(4)
    icp = ICP(max_iter=100, max_dist=2, tol=1e-5)
    icp.set_target(map)
    T_new = icp.fit(scan, init_T=T, verbose=True)
    icp.max_dist = 0.1
    T_new = icp.fit(scan, init_T=T_new, verbose=True)
    R_new, t_new = makeRt(T_new)

    scan_new = transform_points(T_new, scan)

    print(T_new)
    # scan_new = (R_new @ scan.T).T + t_new

    app = q3d.QApplication([])

    # create viewer
    viewer = q3d.Viewer(name='example')
    # add all items to viewer
    viewer.add_items({
        'grid': q3d.GridItem(size=10, spacing=1),
        'map': q3d.CloudItem(size=0.01, alpha=0.5, point_type='SPHERE', \
                              color_mode='FLAT', color='#00ff00', depth_test=True),
        'scan': q3d.CloudItem(size=0.01, alpha=0.5, point_type='SPHERE', \
                              color_mode='FLAT', color='#ff0000', depth_test=True),
        'norm': q3d.LineItem(width=2, color='#00ff00', line_type='LINES')})

    viewer['map'].set_data(map)
    viewer['scan'].set_data(scan_new)

    viewer.show()
    app.exec()

