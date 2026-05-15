"""
Convex trajectory optimization through a safe polytopic corridor.

Given a sequence of overlapping axis-aligned polytopes covering the free
space between start and goal, solve for a polyline path that:
  - Has endpoints fixed at start and goal
  - Stays inside its assigned polytope at every node
  - Minimizes path length plus a small acceleration-smoothness penalty

The problem is a second-order cone program (SOCP). The length term
sum_i ||P[i+1] - P[i]||_2 is a sum of norms (SOCP), the smoothness term
sum_i ||P[i-1] - 2 P[i] + P[i+1]||_2^2 is a quadratic, and corridor
membership is a set of linear inequalities. We solve with Clarabel.
"""

import numpy as np
import cvxpy as cp


def optimize_path_corridor(
    start: tuple[float, float],
    goal: tuple[float, float],
    corridor: list,                   # list of Polytope
    region_assignment: list[int],     # length n_points, value in [0, len(corridor))
    smoothness_weight: float = 0.5,
    d_max: float | None = None,
    solver: str = 'CLARABEL',
    verbose: bool = False,
) -> np.ndarray:
    """Solve the corridor-constrained path optimization.

    Args:
        start: (x, y) starting point. Must lie in corridor[region_assignment[0]].
        goal: (x, y) goal point. Must lie in corridor[region_assignment[-1]].
        corridor: list of Polytope objects (with .A and .b attributes giving
            half-space form A @ x <= b).
        region_assignment: integer list of length n_points. Entry i says which
            polytope point P[i] is constrained to lie in.
        smoothness_weight: lambda on the squared-acceleration smoothness term.
            Small values (0.1 - 1.0) keep the path short while rounding off
            kinks at polytope handoffs. Set to 0 for pure length minimization.
        d_max: optional cap on ||P[i+1] - P[i]||. Helps numerical conditioning
            and keeps point spacing roughly uniform. None means unconstrained.
        solver: CVXPY solver name. Default Clarabel.
        verbose: pass through to the solver.

    Returns:
        (n_points, 2) numpy array of optimized path coordinates.

    Raises:
        ValueError on solver failure or input inconsistencies.
    """
    n = len(region_assignment)
    if n < 2:
        raise ValueError(f"Need at least 2 points; got {n}.")
    if not (0 <= region_assignment[0] < len(corridor)):
        raise ValueError("Bad region_assignment[0].")
    if not (0 <= region_assignment[-1] < len(corridor)):
        raise ValueError("Bad region_assignment[-1].")

    # Validate that start/goal lie in their assigned regions
    r_start = corridor[region_assignment[0]]
    r_goal  = corridor[region_assignment[-1]]
    if not r_start.contains(start):
        raise ValueError(f"Start point {start} not inside its assigned polytope.")
    if not r_goal.contains(goal):
        raise ValueError(f"Goal point {goal} not inside its assigned polytope.")

    # Decision variable: the (n x 2) path
    P = cp.Variable((n, 2))

    # --- Objective ---
    # Length: sum of segment norms. cp.norm(P[1:] - P[:-1], 2, axis=1)
    # returns a vector of per-row norms (length n-1); cp.sum gives total length.
    segments = P[1:] - P[:-1]
    length_term = cp.sum(cp.norm(segments, 2, axis=1))

    # Smoothness: squared discrete acceleration at interior points.
    if n >= 3:
        accel = P[:-2] - 2 * P[1:-1] + P[2:]
        smoothness_term = cp.sum_squares(accel)
        objective = cp.Minimize(length_term + smoothness_weight * smoothness_term)
    else:
        objective = cp.Minimize(length_term)

    # --- Constraints ---
    constraints = []

    # Endpoints pinned
    constraints.append(P[0] == np.asarray(start, dtype=float))
    constraints.append(P[-1] == np.asarray(goal, dtype=float))

    # Corridor membership: each point in its assigned polytope.
    # For axis-aligned polytopes we use bound-form constraints directly,
    # which CVXPY can canonicalize more efficiently than 4 row inequalities.
    for i in range(n):
        poly = corridor[region_assignment[i]]
        constraints.append(P[i, 0] >= poly.x_min)
        constraints.append(P[i, 0] <= poly.x_max)
        constraints.append(P[i, 1] >= poly.y_min)
        constraints.append(P[i, 1] <= poly.y_max)

    # Optional step-size cap
    if d_max is not None:
        for i in range(n - 1):
            constraints.append(cp.norm(P[i + 1] - P[i], 2) <= d_max)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=verbose)

    if prob.status not in ('optimal', 'optimal_inaccurate'):
        raise ValueError(f"Solver returned status {prob.status}")

    return np.asarray(P.value)