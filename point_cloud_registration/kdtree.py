"""
Copyright 2025 Liu Yang
Distributed under MIT license. See LICENSE for more information.
"""

USE_KDTREE = "PYKDTree"  # Set to "PYKDTree", "SMALL_GICP", or "SCIPY" based on your preference

# We find that PyKDTree is faster than SMALL_GICP or SCIPY but comes with an LGPL3.0 license.
# SMALL_GICP is faster than SCIPY, and both have an MIT license.
# By default, we use PYKDTree. If you want to use SMALL_GICP or SCIPY, please set USE_KDTREE to "SMALL_GICP" or "SCIPY".
# Ensure the required libraries are installed:
# pip install pykdtree  # For PYKDTree
# pip install small-gicp  # For SMALL_GICP
# pip install scipy  # For SCIPY
# For performance reasons, we may implement our own KDTree in the future.


if USE_KDTREE == "PYKDTree":
    try:
        from pykdtree.kdtree import KDTree as PyKDTree
        KDTree = PyKDTree
        print("Using PyKDTree for KDTree implementation.")
    except ImportError:
        print("PyKDTree not available.")
        KDTree = None
elif USE_KDTREE == "SMALL_GICP":
    try:
        import numpy as np
        from small_gicp import KdTree as SmallGICPKDTree
        class SmallGICPKDTreeWrapper:
            def __init__(self, data):
                """
                Wrapper for small_gicp.KdTree to provide a consistent interface.
                :param data: numpy array of points to build the KDTree.
                """
                self.num_threads = 8
                self.tree = SmallGICPKDTree(data, self.num_threads)

            def query(self, points, k=1):
                """
                Perform k-nearest neighbor search.
                :param points: numpy array of query points.
                :param k: number of nearest neighbors to find.
                :return: distances and indices of the nearest neighbors.
                """
                if k == 1:
                    indices, distances = self.tree.batch_nearest_neighbor_search(
                        points, self.num_threads)
                elif k > 1:
                    indices, distances = self.tree.batch_knn_search(
                        points, k, self.num_threads)
                return np.array(distances), np.array(indices)
        KDTree = SmallGICPKDTreeWrapper
        print("Using SmallGICP KDTreeWrapper for KDTree implementation.")
    except ImportError:
        print("SmallGICP not available.")
        KDTree = None
elif USE_KDTREE == "SCIPY":
    try:
        from scipy.spatial import cKDTree as SciPyKDTree
        KDTree = SciPyKDTree
        print("Using SciPy cKDTree for KDTree implementation.")
    except ImportError:
        print("SciPy not available.")
        KDTree = None
else:
    print("Invalid KDTree implementation specified.")
    KDTree = None

