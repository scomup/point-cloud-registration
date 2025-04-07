"""
Copyright 2025 Liu Yang
Distributed under MIT license. See LICENSE for more information.

This file originates from scomup/MathematicalRobotics and is licensed under the MIT License.
See the following link for details:
https://github.com/scomup/MathematicalRobotics/blob/main/mathR/utilities/math_tools.py
"""

import numpy as np

epsilon = 1e-5


def huber_weight(r, d=1.0):
    weights = np.ones_like(r)
    mask = r > d
    weights[mask] = d / r[mask]
    return weights


def skew_time_vector(v1, v2):   
    # Efficiently compute skew(v1) @ v2
    x, y, z = v1[:, 0], v1[:, 1], v1[:, 2]
    a, b, c = v2[:, 0], v2[:, 1], v2[:, 2]
    # skew(v1) @ v2
    res = np.zeros((v1.shape[0], 3))
    res[:, 0] = -z * b + y * c
    res[:, 1] = z * a - x * c
    res[:, 2] = -y * a + x * b
    return res


def skews(vectors):
    # Efficiently compute skew-symmetric matrices for a batch of vectors
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    res = np.zeros((vectors.shape[0], 3, 3))
    res[:, 0, 1], res[:, 0, 2] = -z, y
    res[:, 1, 0], res[:, 1, 2] = z, -x
    res[:, 2, 0], res[:, 2, 1] = -y, x
    return res


def skew2(v):
    """
    Efficiently compute sum of skew(v).T @ skew(v)
    """
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    z2 = np.sum(z * z)
    y2 = np.sum(y * y)
    x2 = np.sum(x * x)
    xy = np.sum(x * y)
    xz = np.sum(x * z)
    yz = np.sum(y * z)
    A = np.array([[z2 + y2, -xy, -xz],
                  [-xy, x2 + z2, -yz],
                  [-xz, -yz, x2 + y2]])
    return A


def skew(vector):
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])


def makeT(R, t):
    n = t.shape[0]
    T = np.eye(n+1)
    T[0:n, 0:n] = R
    T[0:n, n] = t
    return T


def makeRt(T):
    n = T.shape[0] - 1
    return T[0:n, 0:n], T[0:n, n]


def expSO3(omega):
    """
    Exponential map of SO3
    see:
    https://github.com/scomup/MathematicalRobotics/blob/main/docs/3d_rotation_group.md
    """
    theta2 = omega.dot(omega)
    theta = np.sqrt(theta2)
    nearZero = theta2 <= epsilon
    W = skew(omega)
    if (nearZero):
        return np.eye(3) + W
    else:
        K = W/theta
        KK = K.dot(K)
        sin_theta = np.sin(theta)
        one_minus_cos = 1 - np.cos(theta)
        R = np.eye(3) + sin_theta * K + one_minus_cos * KK  # rotation.md (10)
        return R


def plus(T, dx):
    """
    define the boxplus operator on SE(3)
    """
    dR = expSO3(dx[3:])
    dt = dx[:3]
    dT = makeT(dR, dt)
    return T @ dT


def transform_points(T, points):
    R, t = makeRt(T)
    return (R @ points.T).T + t


def numerical_derivative(func, param, idx, plus=lambda a, b: a + b, minus=lambda a, b: a - b, delta=1e-5):
    r = func(*param)
    m = r.shape[0]
    n = param[idx].shape[0]
    J = np.zeros([m, n])
    for j in range(n):
        dx = np.zeros(n)
        dx[j] = delta
        param_delta = param.copy()
        param_delta[idx] = plus(param[idx], dx)
        J[:, j] = minus(func(*param_delta), r)/delta
    return J


