"""
optimizer.py — Bubble-constrained SOCP path smoother.

Takes a dense sequence of reference waypoints and their corresponding safety
bubbles, and finds the smoothest path that stays inside those bubbles.

The optimization problem is a Second-Order Cone Program (SOCP):
  - Decision variables: P, a (K x 2) matrix of optimized (x, y) positions
  - Objective: minimize smoothness cost (+ optional closeness-to-reference cost)
  - Constraints:
      * Endpoints are pinned to the A* start/goal exactly
      * Each point must stay within its corresponding bubble radius
      * Consecutive points can't be more than d_max apart (prevents bunching)

SOCPs are convex, so the solver is guaranteed to find the global optimum —
there are no local minima to get stuck in.
"""

import numpy as np
import cvxpy as cp


def optimize_path(q, bubbles, lambda_reg=1.0, d_max=1.5):
    """Solve the bubble-constrained SOCP to find a smooth path.

    Parameters
    ----------
    q : (K, 2) numpy array
        Reference waypoints from A* in world coordinates (already dense from
        resample_by_spacing). These serve as both the bubble centers and the
        anchor that Tikhonov regularization pulls toward.
    bubbles : list of (cx, cy, r) tuples, length K
        Safety bubbles — one per waypoint. Each optimized point P[i] must
        stay within radius bubbles[i][2] of its reference point q[i].
    lambda_reg : float
        Weight on the Tikhonov regularization term.
          0.0 = pure rubber-band: path is free to go anywhere in the bubbles,
                maximally smooth but may drift from the intended route.
          large = path stays close to the A* reference, less smoothing occurs.
    d_max : float
        Maximum allowed Euclidean distance between consecutive optimized points.
        Prevents all interior points from bunching near the endpoints while a
        large gap forms in the middle.

    Returns
    -------
    (K, 2) numpy array of optimized waypoint coordinates.
    """
    q = np.asarray(q, dtype=float)
    K = q.shape[0]   # number of waypoints / decision variables

    # Every waypoint needs exactly one bubble
    assert len(bubbles) == K, f"Mismatch: {K} waypoints but {len(bubbles)} bubbles."

    # ---- Decision Variables ----
    # P[i] is the optimized (x, y) position of the i-th waypoint.
    # Shape (K, 2): K rows (one per point), 2 columns (x and y).
    P = cp.Variable((K, 2))

    # ---- Objective: Smoothness (Acceleration) Term ----
    # The discrete second derivative at interior point i is:
    #   a_i = P[i-1] - 2*P[i] + P[i+1]
    # This is the "acceleration" — how sharply the path bends at point i.
    # Minimizing the sum of squared accelerations produces a path that turns
    # as gradually as possible, giving smooth curves instead of sharp kinks.
    #
    # Vectorized over all K-2 interior points at once:
    #   P[:-2]  = points 0 ... K-3  (the "previous" points for each interior node)
    #   P[1:-1] = points 1 ... K-2  (the interior points being penalized)
    #   P[2:]   = points 2 ... K-1  (the "next" points for each interior node)
    accel_cost = cp.sum_squares(P[:-2] - 2 * P[1:-1] + P[2:])

    # ---- Objective: Tikhonov Regularization Term ----
    # Pulls each optimized point toward its corresponding A* reference point q[i].
    # Without this (lambda_reg=0), the path acts like a rubber band stretched
    # between the fixed endpoints — maximally smooth but potentially far from
    # the original route. With a large lambda_reg, smoothing is suppressed.
    tikhonov_cost = cp.sum_squares(P - q)

    # Combined objective: trade off smoothness against closeness to the reference
    objective = cp.Minimize(accel_cost + lambda_reg * tikhonov_cost)

    # ---- Constraints ----
    constraints = []

    # 1. Endpoint anchoring: start and goal must exactly match the A* endpoints.
    constraints += [
        P[0]  == q[0],
        P[-1] == q[-1],
    ]

    # 2. Bubble constraints: each point must stay within its safety bubble.
    #    norm(P[i] - q[i], 2) <= r[i] is a second-order cone constraint.
    #    The bubble radius was computed as the distance from q[i] to the nearest
    #    obstacle, so any P[i] inside the bubble is guaranteed collision-free.
    for i in range(K):
        r = bubbles[i][2]   # radius of this waypoint's bubble
        constraints.append(cp.norm(P[i] - q[i], 2) <= r)

    # 3. Step-size cap: consecutive optimized points can't be more than d_max apart.
    #    Without this, the rubber-band effect can cause all interior points to
    #    collapse toward the path center, leaving the endpoints densely sampled
    #    and a large unconstrained jump in the middle.
    for i in range(K - 1):
        constraints.append(cp.norm(P[i + 1] - P[i], 2) <= d_max)

    # ---- Solve ----
    prob = cp.Problem(objective, constraints)
    # CLARABEL is a modern interior-point solver well-suited to SOCPs.
    # It finds the exact global optimum because the problem is convex.
    prob.solve(solver=cp.CLARABEL)

    # Check that the solver converged to a valid solution
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(
            f"Solver failed to find an optimal solution. Status: {prob.status}"
        )

    return P.value   # (K, 2) numpy array of optimized coordinates
