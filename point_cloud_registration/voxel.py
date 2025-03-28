import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree


class VoxelCell:
    """
    Store the normal distribution information of a voxel cell,
    used for Normal Distributions Transform and Point-to-Plane ICP.
    """
    def __init__(self):
        self.num = 0
        self.sum = np.zeros(3)
        self.ppt = np.zeros([3, 3])
        self.mean = np.zeros(3)
        self.cov = np.zeros([3, 3])
        self.norm = np.zeros(3)

    def add_points(self, points):
        self.num += points.shape[0]
        self.sum += np.sum(points, axis=0)
        self.ppt += np.dot(points.T, points)
        self.mean = self.sum / self.num
        self.cov = (self.ppt - 2 * np.outer(self.sum, self.mean)) / \
            self.num + np.outer(self.mean, self.mean)
        _, eigenvectors = np.linalg.eigh(self.cov)
        self.norm = eigenvectors[:, 0]

    def set_points(self, points):
        self.num = points.shape[0]
        self.sum = np.sum(points, axis=0)
        self.ppt = np.dot(points.T, points)
        self.mean = self.sum / self.num
        self.cov = (self.ppt - 2 * np.outer(self.sum, self.mean)) / \
            self.num + np.outer(self.mean, self.mean)
        _, eigenvectors = np.linalg.eigh(self.cov)
        self.norm = eigenvectors[:, 0]


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
        # Compute keys for all points
        keys = self.get_keys(points, self.voxel_size)
        # Find unique keys and their indices
        voxel_points = []
        self.voxel_keys = []
        unique_keys, inverse_indices = np.unique(keys, return_inverse=True)
        for i, key in enumerate(unique_keys):
            pts = points[inverse_indices == i]
            # do not add the voxel with less than 10 points
            if len(pts) < 4:
                continue
            self.voxels[key].add_points(pts)
            voxel_points.append(self.voxels[key].mean)
            self.voxel_keys.append(key)
        self.kdtree = cKDTree(np.array(voxel_points))

    def find(self, point):
        # usd kdtree to find the nearest cell
        _, idx = self.kdtree.query(point)
        key = list(self.voxels.keys())[idx]
        return self.voxels[key]


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
    try:
        import q3dviewer as q3d
    except ImportError:
        print("Please install q3dviewer using 'pip install q3dviewer'")
        exit(0)

    points, _ = q3d.load_pcd("/home/liu/tmp/recorded_frames/clouds/1.pcd")
    voxel_size = 0.5
    voxels = VoxelGrid(voxel_size)
    import time
    t1 = time.time()
    voxels.add_points(points['xyz'])
    t2 = time.time()
    print(f"add points time: {t2 - t1}")
    app = q3d.QApplication([])

    # create viewer
    viewer = q3d.Viewer(name='example')
    # add all items to viewer
    viewer.add_items({
        'grid': q3d.GridItem(size=10, spacing=1),
        'scan': q3d.CloudItem(size=0.01, alpha=1, point_type='SPHERE', depth_test=True),
        'norm': q3d.LineItem(width=2, color='#00ff00', line_type='LINES')})
    
    color_points = color_by_voxel(points['xyz'], voxel_size)
    viewer['scan'].set_data(color_points)

    lines = []
    for key, cell in voxels.voxels.items():
            lines.append(cell.mean)
            lines.append(cell.mean + cell.norm * 0.1)
    lines = np.array(lines)
    viewer['norm'].set_data(lines)

    viewer.show()
    app.exec()
    """
    """""


