import numpy as np
from point_cloud_registration.math_tools import plus


class Registration:
    def __init__(self, max_iter=100, tol=1e-6):
        """
        Base class for point cloud registration methods.
        :param max_iter: Maximum number of iterations.
        :param tol: Convergence tolerance.
        """
        self.max_iter = max_iter
        self.tol = tol
        self.target = None

    def set_target(self, target):
        """
        Set the target point cloud.
        :param target: Target point cloud (Nx3 array).
        """
        self.target = target
        raise NotImplementedError("set_target is not implemented.")

    def update_target(self, target):
        """
        Update the target point cloud. 
        This method is useful to re-use the same target point cloud
        :param target: Target point cloud (Nx3 array).
        """
        self.target = target
        raise NotImplementedError("update_target is not implemented.")

    def linearize(self, cur_T, source):
        """
        Linearize the objective function and compute the Jacobian and residual.
        :param
        cur_T: Current transformation (4x4 array).
        source: Source point cloud (Nx3 array).
        :return: Jacobian (Nx6 array), residual (Nx3 array), weights (N array).
        """
        raise NotImplementedError("linearize is not implemented.")


    def fit(self, source, init_T=np.eye(4), verbose=False):
        """
        use Gauss-Newton method to find the transformation 
        that aligns the source point cloud to the target point cloud.
        :param source: Source point cloud (Nx3 array).
        :param init_T: Initial transformation (4x4 array).
        :param verbose: Print error at each iteration.
        :return: Final transformation (4x4 array).
        """
        cur_T = init_T
        best_T = cur_T
        best_error = np.inf
        for i in range(self.max_iter):
            Js, rs, weights = self.linearize(cur_T, source)
            # Gauss-Newton step
            """
            Js, rs and weights is [N, M, 6], [N, M], [N] array respectively
            where N is the number of points, M is the dimension of the residual
            6: 3 for SO(3), 3 for translation
            H = sum (Js[i].T @ Js[i] * weights[i])
            g = sum (Js[i].T @ rs[i] * weights[i])
            e2 = sum (rs[i] * rs[i] * weights[i])
            """
            # use einsum to parallelize the computation
            JsT = Js.transpose(0, 2, 1)
            H = np.einsum('nij,njk,n->ik', JsT, Js, weights)
            g = np.einsum('nij,nj,n->i', JsT, rs, weights)
            e2 = np.einsum('ni,ni,n->', rs, rs, weights)

            # ensure the error is decreasing
            if e2 < best_error:
                best_error = e2
                best_T = cur_T.copy()
            else:
                break

            if verbose:
                print(f"iter {i}, error {e2}")
            
            # solve the linear system
            dx = -np.linalg.solve(H, g)
    
            # check convergence
            if np.linalg.norm(dx) < self.tol:
                break

            # Update transformation
            cur_T = plus(cur_T, dx)

        return best_T
