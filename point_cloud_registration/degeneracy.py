"""
Degeneracy detection and solution remapping for point cloud registration.

Least-squares registration (ICP and its variants) takes a Gauss-Newton step
``dx = -H⁻¹ g`` at every iteration, where ``H`` is the 6×6 approximate Hessian
(``JᵀJ``) accumulated over the correspondences and ``g`` is the gradient.  When
the scene is geometrically under-determined — a flat plane, a straight
corridor, a surface of revolution — ``H`` is rank-deficient or ill-conditioned
along one or more directions, and the unconstrained solve slides the estimate
along those directions driven by nothing but noise.

This module implements *solution remapping*: eigendecompose ``H``, classify each
eigen-direction as constrained or degenerate, and project the update so that its
component along every degenerate direction is zero.  The estimate is then only
updated where the data actually supports it, and stays put elsewhere.

Scope: :func:`analyse_hessian` *classifies* directions and copes with an
exactly rank-deficient ``H``; :func:`apply_sr_solve` still inverts ``H`` to form
the step, so it requires an invertible one.  Damping a singular Hessian is the
caller's responsibility — see :func:`apply_sr_solve` for the contract.

References:
    A. Hinduja, B.-J. Ho and M. Kaess, "Degeneracy-Aware Factors with
    Applications to Underwater SLAM", IEEE/RSJ International Conference on
    Intelligent Robots and Systems (IROS), 2019, pp. 1293-1299,
    doi: 10.1109/IROS40897.2019.8968577 — introduced solution remapping
    inside the ICP iteration itself (its Algorithm 1 is the procedure
    implemented by :func:`analyse_hessian` + :func:`apply_sr_solve`,
    including the ``sqrt(lambda_max / lambda_min)`` threshold).

    J. Zhang, M. Kaess and S. Singh, "On Degeneracy of Optimization-based
    State Estimation Problems", IEEE International Conference on Robotics
    and Automation (ICRA), 2016 — the origin of the solution-remapping
    update for optimization-based state estimation.

DOF order
---------
All 6-vectors and 6×6 matrices here use this library's tangent-space ordering::

    [tx, ty, tz, ωx, ωy, ωz]

i.e. translation first, rotation second — the ordering consumed by
``math_tools.plus()`` and produced by the ``calc_H_g_e2()`` methods of the
registration classes.  These are right-tangent (body-frame) increments: the
step is applied as ``T @ [expSO3(dx[3:]) | dx[:3]]``, so the eigenvectors of
``H`` — and therefore the directions this module classifies and zeroes — are
body-frame directions of the current estimate.  Callers that work in a
different convention (for example a factor-graph library that orders rotation
first, or reasons in world-frame axes) must permute or rotate ``H`` and ``g``
before calling and map the returned step back afterwards.

This module depends on NumPy only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DegeneracyResult:
    """
    Output of analyse_hessian().

    Attributes
    ----------
    eigenvalues:
        Ascending eigenvalues of H.
    eigenvectors:
        Corresponding eigenvectors as *columns*, so ``eigenvectors[:, i]`` is the
        unit direction whose curvature is ``eigenvalues[i]``.
    condition_number:
        ``sqrt(λ_max / λ_min)``; also the adaptive degeneracy threshold when no
        explicit threshold was supplied.  ``inf`` for a rank-zero Hessian.
    lambda_threshold:
        The eigenvalue threshold actually applied.
    degenerate_mask:
        ``True`` where the direction is degenerate (data does not constrain it).
    num_constrained_dof:
        Number of directions the data does constrain, 0-6.
    is_degenerate:
        ``True`` if at least one direction is degenerate.
    V_constrained:
        The constrained eigenvectors as unit-norm *rows*.
    """
    eigenvalues: np.ndarray       # (6,) ascending, in PCR order
    eigenvectors: np.ndarray      # (6,6) columns are eigenvectors, in PCR order
    condition_number: float       # sqrt(λ_max / λ_min)
    lambda_threshold: float       # eigenvalue threshold used
    degenerate_mask: np.ndarray   # (6,) bool, True = degenerate direction
    num_constrained_dof: int      # count of non-degenerate directions
    is_degenerate: bool           # True if any DOF is degenerate
    # Non-degenerate eigenvector rows in PCR order, shape (num_constrained_dof, 6).
    # Empty (0, 6) array when all directions are degenerate.
    V_constrained: np.ndarray


def analyse_hessian(
    H: np.ndarray,
    lambda_threshold: Optional[float] = None,
) -> DegeneracyResult:
    """
    Eigendecompose a 6×6 registration Hessian (ATA) and identify degenerate
    directions.

    H must be in this library's DOF order: [tx, ty, tz, ωx, ωy, ωz].

    Criterion
    ---------
    A direction is degenerate when its eigenvalue falls below the threshold

        lambda_cn = sqrt(lambda_max / lambda_min)

    the square root of the Hessian's condition number.  Note what that compares:
    an eigenvalue, which carries the scale of the problem (correspondence count,
    units of the residual), against a *unitless* ratio.  Classification therefore
    depends on the absolute magnitude of H and not on its conditioning alone —
    scaling H uniformly moves the eigenvalues but leaves lambda_cn untouched.
    ``H = 2*I`` and ``H = 0.5*I`` are both perfectly conditioned (lambda_cn = 1),
    yet the first has all six directions constrained and the second has all six
    degenerate.  Concretely, the best-observed direction survives only when
    ``lambda_max >= sqrt(lambda_max / lambda_min)``, i.e. when
    ``lambda_max * lambda_min >= 1``.

    This is the criterion as published and as implemented in the reference
    solution-remapping code, and it is kept here unchanged for fidelity to it.
    Callers who have a calibrated absolute curvature floor for their sensor and
    geometry should use the lambda_threshold override instead of relying on it.

    Structural zeros are handled in two phases so that exact rank deficiency does
    not swamp the test:

    1. Eigenvalues below ``1e-10 * lambda_max`` are treated as structural zeros
       (a genuinely unobserved direction: too few correspondences, or a scene
       with an exact continuous symmetry).  They are always degenerate.
    2. lambda_cn is then formed from the smallest *non*-structural-zero
       eigenvalue.  Without this, a single exact-zero eigenvalue would send the
       condition number to infinity and mark every direction — including the
       well-observed ones — degenerate.

    A Hessian whose largest eigenvalue is itself below ``1e-10`` carries no
    information at all (e.g. zero correspondences); every direction is reported
    degenerate and ``V_constrained`` is an empty ``(0, 6)`` array.

    This classification is well defined for a singular H, but classifying is not
    solving: :func:`apply_sr_solve` inverts H and will raise on an exactly
    singular one.  See its docstring for how to pair the two.

    If lambda_threshold is not None it overrides the adaptive threshold with a
    fixed absolute eigenvalue floor (useful when a caller has a calibrated
    minimum curvature for its sensor and geometry).  The reported condition
    number is still computed, with lambda_min floored at ``1e-12`` to keep it
    finite.
    """
    H = np.asarray(H, dtype=float)
    assert H.shape == (6, 6), f"Expected 6×6 Hessian, got {H.shape}"

    eigenvalues, eigenvectors = np.linalg.eigh(H)  # ascending order
    lam_max = float(eigenvalues[-1])

    # Zero or near-zero Hessian → no correspondences → all directions degenerate.
    if lam_max < 1e-10:
        return DegeneracyResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            condition_number=float('inf'),
            lambda_threshold=float('inf'),
            degenerate_mask=np.ones(6, dtype=bool),
            num_constrained_dof=0,
            is_degenerate=True,
            V_constrained=np.empty((0, 6)),
        )

    if lambda_threshold is not None:
        # User-supplied override (e.g. from SR solve fixed threshold).
        lam_min = float(max(eigenvalues[0], 1e-12))
        condition_number = float(np.sqrt(lam_max / lam_min))
        threshold = float(lambda_threshold)
        degenerate_mask = eigenvalues < threshold
    else:
        # Two-phase: structural zeros first, then the condition-number test
        # among the non-zeros.  Rank-deficient Hessians (flat surface,
        # single-point) have exact-zero eigenvalues that must not inflate the
        # condition number to infinity.
        abs_zero_thresh = 1e-10 * lam_max
        struct_zero_mask = eigenvalues < abs_zero_thresh
        if struct_zero_mask.all():
            condition_number = float('inf')
            threshold = float('inf')
            degenerate_mask = np.ones(6, dtype=bool)
        else:
            lam_min_pos = float(eigenvalues[~struct_zero_mask][0])
            condition_number = float(np.sqrt(lam_max / lam_min_pos))
            threshold = condition_number
            # Structural zeros are degenerate no matter where the adaptive
            # threshold lands: with a tight non-zero spectrum the threshold
            # can fall below the structural-zero cutoff (e.g. five equal
            # large eigenvalues give threshold 1), and the phase-1 verdict
            # must not be overwritten by the phase-2 test.
            degenerate_mask = (eigenvalues < threshold) | struct_zero_mask

    constrained_indices = np.where(~degenerate_mask)[0]
    num_constrained = int(len(constrained_indices))

    # V_constrained: rows are non-degenerate eigenvectors in PCR order
    V_constrained = eigenvectors[:, constrained_indices].T  # (num_constrained, 6)

    return DegeneracyResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        condition_number=condition_number,
        lambda_threshold=threshold,
        degenerate_mask=degenerate_mask,
        num_constrained_dof=num_constrained,
        is_degenerate=bool(np.any(degenerate_mask)),
        V_constrained=V_constrained,
    )


def apply_sr_solve(
    H: np.ndarray,
    g: np.ndarray,
    deg: DegeneracyResult,
) -> np.ndarray:
    """
    Apply solution-remapping to the registration linear solve.

    Standard solve:  dx = -H⁻¹ g
    SR solve:        dx = -Vf_inv @ Vu_filtered @ H⁻¹ @ g

    ``Vu_filtered`` is the matrix of eigenvector rows with the degenerate rows
    replaced by zero, and ``Vf_inv`` maps back from the eigenbasis to the DOF
    basis.  The composition projects the Gauss-Newton step onto the span of the
    constrained eigen-directions: the optimizer still takes the full step where
    the data supports it, and no step at all along the ill-conditioned DOFs.

    When nothing is degenerate this reduces exactly to the standard solve.

    Contract: H must be invertible.  Detecting degeneracy does not remove the
    inversion — ``np.linalg.solve(H, g)`` runs on the H you pass, before any
    projection, so an exactly singular H raises ``np.linalg.LinAlgError`` no
    matter what ``deg`` says about it.  A caller whose raw Hessian can be exactly
    rank-deficient (a plane fitted to coplanar points, fewer correspondences than
    DOFs) should pass a regularized ``H + lambda*I`` here while taking ``deg``
    from :func:`analyse_hessian` on the **raw** H.  An isotropic shift adds
    lambda to every eigenvalue and leaves the eigenvectors untouched, so the
    projection this function applies is unchanged; only the step length along the
    constrained directions is damped, and analysing the raw H keeps the
    degeneracy classification honest (damping before analysis raises lambda_min
    and so lowers the threshold; for a large enough lambda the weak directions
    stop being flagged at all).

    Args:
        H:   6×6 Hessian from calc_H_g_e2 (in PCR order), invertible — see Contract
        g:   6-element gradient from calc_H_g_e2
        deg: DegeneracyResult from analyse_hessian(H)

    Returns:
        6-element step dx in PCR order

    Raises:
        numpy.linalg.LinAlgError: if H (or the eigenvector matrix) is singular.
    """
    Vu = deg.eigenvectors.T.copy()          # (6, 6) rows = eigenvectors
    for i in range(6):
        if deg.degenerate_mask[i]:
            Vu[i, :] = 0.0                  # zero degenerate rows

    Vf_inv = np.linalg.inv(deg.eigenvectors.T)
    H_inv_g = np.linalg.solve(H, g)
    return -(Vf_inv @ Vu @ H_inv_g)
