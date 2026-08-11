"""
Copyright 2025 Liu Yang
Distributed under MIT license. See LICENSE for more information.
"""

import numpy as np
from point_cloud_registration.math_tools import plus
from point_cloud_registration.degeneracy import analyse_hessian, apply_sr_solve


class Registration:
    def __init__(self, max_iter=30, tol=1e-3):
        """
        Base class for point cloud registration methods.
        :param max_iter: Maximum number of iterations.
        :param tol: Convergence tolerance.
        """
        self.max_iter = max_iter
        self.tol = tol
        self._is_target_set = False
        self._last_hessian = None

    def is_target_set(self):
        """
        Check if the target point cloud is set.
        :return: True if target is set, False otherwise.
        """
        return self._is_target_set

    def set_target(self, target):
        """
        Set the target point cloud.
        :param target: Target point cloud (Nx3 array).
        """
        self._is_target_set = True
        raise NotImplementedError("set_target is not implemented.")

    def update_target(self, target):
        """
        Update the target map continuously.
        It is very useful for saving time by avoiding rebuild the KDTree or 3D voxels.
        But we do not plan to implement it currently.
        :param target: Target point cloud (Nx3 array).
        """
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
    
    def calc_H_g_e2(self, cur_T, source, dx_norm=np.inf):
        """
        Compute the Hessian, gradient, and squared error.
        :param cur_T: Current transformation (4x4 array).
        source: Source point cloud (Nx3 array).
        :return: Hessian (6x6 array), gradient (6 array), squared error (scalar).
        """
        Js, rs, weights = self.linearize(cur_T, source)
        # use einsum to parallelize the computation
        JsT = Js.transpose(0, 2, 1)
        H = np.einsum('nij,njk,n->ik', JsT, Js, weights)
        g = np.einsum('nij,nj,n->i', JsT, rs, weights)
        e2 = np.einsum('ni,ni,n->', rs, rs, weights)
        return H, g, e2


    def align(self, source, init_T=np.eye(4), verbose=False,
              use_solution_remapping=False, sr_lambda_threshold=None,
              lm_damping=False):
        """
        Gauss-Newton alignment of the source cloud onto the target.

        The per-iteration step dx is a body-frame right-tangent increment
        applied as T @ [expSO3(dx[3:]) | dx[:3]] (see math_tools.plus).

        :param source: Source point cloud (Nx3 array).
        :param init_T: Initial transformation (4x4 array).
        :param verbose: Print error at each iteration.
        :param use_solution_remapping: If True, zero the step in degenerate
            Hessian eigenvector directions at each iteration (SR mode;
            Hinduja, Ho & Kaess, IROS 2019, Algorithm 1 — solution remapping
            per Zhang, Kaess & Singh, ICRA 2016; see degeneracy.py for full
            references).
        :param sr_lambda_threshold: Override the condition-number threshold
            for SR. None = adaptive (sqrt(lambda_max / lambda_min)).
        :param lm_damping: If True, solve the damped system (H + lambda*I) dx = -g
            instead of H dx = -g, with lambda scaled to the trace of H
            (Levenberg-Marquardt). This keeps the linear solve well posed on
            geometry that leaves H singular or near-singular — a single flat
            surface, a straight corridor, too few correspondences — where the
            plain solve raises numpy.linalg.LinAlgError or returns a step
            dominated by noise. On well-conditioned data the damping is small
            enough to leave the solution unchanged. Pair it with
            use_solution_remapping when the Hessian may be exactly rank
            deficient: the degeneracy analysis runs on the raw H so the
            classification stays honest, while the solve uses the damped one.
        :return: Final transformation (4x4 array).
        """
        if self.is_target_set() is False:
            raise ValueError("Target is not set.")

        source = source.astype(np.float32)
        # Copy: align() must not return the caller's array (or the shared
        # mutable np.eye(4) default) when it converges before the first step.
        cur_T = init_T.copy()
        H_final = None
        converged = False

        for i in range(self.max_iter):
            H, g, e2 = self.calc_H_g_e2(cur_T, source)
            H_final = H
            if verbose:
                print(f"iter {i}, error {e2}")

            if lm_damping:
                trace_H = np.trace(H)
                lambda_lm = max(1e-4 * trace_H / 6.0 if trace_H > 0 else 1e-3, 1e-6)
                H_solve = H + lambda_lm * np.eye(6)
            else:
                H_solve = H

            if use_solution_remapping:
                # Eigendecompose the raw H (correct degeneracy thresholding) but
                # solve on H_solve (numerical stability): an isotropic +lambda*I
                # shift leaves the eigenvectors identical.
                deg = analyse_hessian(H.astype(float), lambda_threshold=sr_lambda_threshold)
                dx = apply_sr_solve(H_solve.astype(float), g.astype(float), deg)
            elif lm_damping:
                try:
                    dx = -np.linalg.solve(H_solve, g)
                except np.linalg.LinAlgError:
                    dx = -np.linalg.lstsq(H_solve, g, rcond=None)[0]
            else:
                dx = -np.linalg.solve(H, g)

            # check convergence
            dx_norm = np.linalg.norm(dx)
            if dx_norm < self.tol:
                converged = True
                break

            # Update transformation
            cur_T = plus(cur_T, dx)

        # If max_iter was exhausted, cur_T advanced past the last
        # linearization, so recompute the Hessian at the returned pose.
        if not converged and H_final is not None:
            H_final, _, _ = self.calc_H_g_e2(cur_T, source)
        self._last_hessian = H_final
        return cur_T

    @property
    def last_hessian(self):
        """
        The 6x6 Gauss-Newton Hessian (J^T W J) evaluated at the pose returned
        by the most recent align() call, or None before any align().

        It is expressed in the right-tangent (body) frame of that pose, DOF
        order [tx, ty, tz, wx, wy, wz] — the increment coordinates consumed by
        math_tools.plus().  Consumers that reason about world-frame axes must
        map directions through the returned pose: a body increment
        [dt, w] moves the estimate by R @ dt (translation) and R @ w
        (rotation axis) in the world; a full left-tangent (world-frame)
        Hessian additionally carries the translation-rotation coupling and
        is obtained with the SE(3) adjoint of the pose,
        H_world = Ad^{-T} @ H @ Ad^{-1}.
        """
        return self._last_hessian