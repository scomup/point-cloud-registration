import numpy as np


class Registration:
    def __init__(self, max_iter=100, tol=1e-6):
        """
        Base class for point cloud registration methods.
        :param max_iter: Maximum number of iterations.
        :param tol: Convergence tolerance.
        """
        self.max_iter = max_iter
        self.tol = tol

    def set_target(self, target):
        """
        Set the target point cloud.
        :param target: Target point cloud (Nx3 array).
        """
        raise NotImplementedError("set_target is not implemented.")

    def update_target(self, target):
        """
        Update the target point cloud. 
        This method is useful to re-use the same target point cloud
        :param target: Target point cloud (Nx3 array).
        """
        raise NotImplementedError("update_target is not implemented.")

    def fit(self, source, init_T=np.eye(4)):
        """
        Fit the source point cloud to the target point cloud.
        :param source: Source point cloud (Nx3 array).
        :param init_T: Initial transformation matrix (4x4 array).
        :return: Final transformation matrix (4x4 array).
        """
        raise NotImplementedError("fit is not implemented.")
