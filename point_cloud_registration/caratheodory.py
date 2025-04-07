"""
TODO (liu): Try using Caratheodory's theorem to select a coreset of points to speed up VPlaneICP or PlaneICP.
This file is my Python implementation of Caratheodory's theorem-based algorithms for exact point 
cloud downsampling.

Current problems:
1. May not work for extracted point sets with small sizes.
2. Only works when the error is very small. May need a better algorithm to decide when to use it.

Based on:
K. Koide, "Exact Point Cloud Downsampling for Fast and Accurate Global Trajectory Optimization,"
arXiv preprint arXiv:2307.02948v3, 2023. https://arxiv.org/abs/2307.02948
The original C++ implementation can be found at:
https://github.com/koide3/caratheodory2/blob/main/src/caratheodory.cpp

This module includes:
- Caratheodory's algorithm for finding exact coresets.
- Fast Caratheodory's algorithm for efficient coreset extraction.
- Utility functions for creating point sets from Jacobians and residuals.
"""

import numpy as np

def null_space(P, tol=1e-12):
    """
    find a vector v s.t. P @ v = 0 and np.sum(v) = 0
    """
    A = P[:, 1:] - P[:, [0]]
    _, s, Vh = np.linalg.svd(A)
    null_mask = np.append(s <= tol, [True] * (Vh.shape[0] - len(s)))
    null_space = Vh[null_mask].T
    v = null_space[:, -1]
    v = np.insert(v, 0, -v.sum())
    return v

def caratheodory(P, u, N_target):
    """
    Caratheodory's algorithm for finding a exact coreset 
    with N_target elements from a weighted input set.
    """
    N = P.shape[1]
    if N <= N_target:
        return P, u, np.ones(N, dtype=int)
    selected_idxs = np.arange(N)
    while P.shape[1] > N_target:
        # Find a v to s.t. P @ v = 0 and np.sum(v) = 0
        v = null_space(P)
        # Find the minimum alpha s.t. u[idx] - alpha * v[idx] == 0
        # for minimum the adjustment of the weight
        alphas = u / v
        idx = np.argmin(np.abs(alphas))
        alpha = alphas[idx]
        w = u - alpha * v

        # remove the point with the weight equal to zero
        P = np.delete(P, idx, axis=1)
        u = np.delete(w, idx)
        selected_idxs = np.delete(selected_idxs, idx)

    return P, u, selected_idxs

def fast_caratheodory(P, u, k, N_target):
    """
    Fast Caratheodory's algorithm for finding a exact coreset
    by divide the original set into k subsets. 
    """
    cur_N = P.shape[1]
    if cur_N <= N_target:
        return P, u, np.arange(cur_N)

    selected_idxs = np.arange(cur_N)

    n_loop = 0
    while cur_N > N_target:
        k = min(k, cur_N)
        indices = np.linspace(0, cur_N, k+1, dtype=int)
        ranges = np.zeros([k, 2], dtype=int)
        ranges[:, 0] = indices[:-1]
        ranges[:, 1] = indices[1:]
        sub_size = ranges[:, 1] - ranges[:, 0]

        # divide the set into k subsets
        P_sub = np.zeros((P.shape[0], k))
        u_sub = np.zeros(k)

        # compute the mean of weights and the mean of the points in each subset
        for i in range(k):
            begin, end = ranges[i]
            u_sub[i] = u[begin:end].sum()
            P_sub[:, i] = (u[begin:end] @ P[:, begin:end].T) / u_sub[i]

        # adjust the number of points in each subset
        N_sub = P.shape[0] + 1
        max_cluster_size = np.max(sub_size)
        if N_sub * max_cluster_size < N_target:
            N_sub = N_target // max_cluster_size

        _, w_sub, selected_sub = caratheodory(P_sub, u_sub, N_sub)

        # combine the selected subsets to a new set
        # adjust the weights of the new set using the weights of the selected subsets
        selected_ranges = ranges[selected_sub]
        selected_indices = np.concatenate(
            [np.arange(begin, end) for begin, end in selected_ranges])
        sum_weights = u_sub[selected_sub]
        w_factors = np.repeat(w_sub / sum_weights, sub_size[selected_sub])

        # remove the points which are out of the selected subsets
        P = P[:, selected_indices]
        u = w_factors * u[selected_indices]
        selected_idxs = selected_idxs[selected_indices]
        cur_N = P.shape[1]
        n_loop += 1
    # print(f"n_loop={n_loop}")

    return P, u, selected_idxs

def create_gn_set(J, r):
    """
    Create a set of points from the Jacobian and the residuals. 
    Use this set to extract a coreset while ensuring that the Hessian,
    gradient, and squared error remain the same as the original.
    """
    N, D = J.shape
    # the upper triangular part size of the Hessian matrix
    NH = D * (D + 1) // 2
    M = NH + D + 1
    # Initialize P
    P = np.empty((N, M))
    # Compute the Hessian matrix
    H = np.einsum('ij,ik->ijk', J, J)
    triu_indices = np.triu_indices(D)
    P[:,:NH] = H[:, triu_indices[0], triu_indices[1]]
    # Compute the gradient
    P[:,NH:NH + D] = J * r[:, np.newaxis]
    # Compute the squared error
    P[:,NH + D] = r ** 2
    return P.T
