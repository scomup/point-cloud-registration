import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from caratheodory import fast_caratheodory, create_gn_set

def test_exact_same_results():
    N = 30000  # Number of input residuals
    k = 64     # Number of clusters for Fast-Caratheodory
    N_target = 128

    J = np.random.randn(N, 6)  # Jacobian
    r = np.random.randn(N)     # residuals

    # t0 = time.time()
    P = create_gn_set(J, r)
    assert N_target > (P.shape[0] + 1), \
        f"N_target size ({N_target}) must be greater than ({P.shape[0] + 1})"
    u = np.ones(P.shape[1])

    _, w, indices = fast_caratheodory(P, u, k, N_target)
    # t1 = time.time()

    H = J.T @ J   # Hessian
    g = J.T @ r   # Gradient
    e2 = r.T @ r  # Squared error

    J_sub = J[indices]
    r_sub = r[indices]
    # equal to S.T @ np.diag(w) @ S but faster
    H_tilde = J_sub.T @ (w[:, np.newaxis] * J_sub)
    g_tilde = J_sub.T @ (w * r_sub)
    e2_tilde = r_sub.T @ (w * r_sub)

    error_H = np.max(np.abs(H - H_tilde))
    error_g = np.max(np.abs(g - g_tilde))
    error_c = np.abs(e2 - e2_tilde)
    error = max(error_H, error_g, error_c)

    # run_time = t1 - t0
    # print(f"error: {error:0.12f}  size: {len(w)}, time:{run_time * 1e3:0.1f} [ms]")

    assert error <= 1e-10, f"Error too large: {error}"

def test_weights_positive():
    N = 30000  # Number of input residuals
    k = 64     # Number of clusters for Fast-Caratheodory
    N_target = 128

    J = np.random.randn(N, 6)  # Jacobian
    r = np.random.randn(N)     # residuals

    P = create_gn_set(J, r)
    u = np.ones(P.shape[1])

    _, w, _ = fast_caratheodory(P, u, k, N_target)

    # Ensure all weights are positive
    assert np.all(len(w) <= N_target), "the size of the exacted set is too large"
    assert np.all(w > 0), "Some weights are not greater than 0"
