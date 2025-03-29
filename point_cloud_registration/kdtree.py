try:
    from pykdtree.kdtree import KDTree as PyKDTree
    KDTree = PyKDTree
    print("Using PyKDTree for KDTree implementation.")
except ImportError:
    from scipy.spatial import cKDTree as SciPyKDTree
    KDTree = SciPyKDTree
    print("Using SciPy cKDTree for KDTree implementation.")
