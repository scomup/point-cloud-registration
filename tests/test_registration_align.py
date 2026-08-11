"""
Behavioral tests for Registration.align().
"""
import numpy as np

from point_cloud_registration.icp import ICP


def test_align_does_not_alias_init_T():
    """
    align() must return a transform the caller owns.

    With source == target the very first step is ~zero, so align()
    converges before ever calling plus(): without the defensive copy it
    returns the init_T object itself — and with the mutable np.eye(4)
    default, a caller mutating the result silently corrupts the default
    for every subsequent align() call in the process.
    """
    np.random.seed(1)
    target = np.random.rand(100, 3)
    icp = ICP(max_iter=10, max_dist=2.0, tol=1e-3)
    icp.set_target(target)
    source = target.astype(np.float32)

    init_T = np.eye(4)
    T = icp.align(source, init_T=init_T)
    assert T is not init_T

    # Mutating the result must not corrupt the shared default argument.
    T_default = icp.align(source)
    T_default[0, 3] = 123.0
    T_again = icp.align(source)
    np.testing.assert_allclose(T_again, np.eye(4), atol=1e-6)
