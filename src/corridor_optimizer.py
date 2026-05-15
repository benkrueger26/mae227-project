"""
corridor_optimizer.py — Polytope-corridor-constrained SOCP path optimizer.

Given a sequence of overlapping axis-aligned polytopes (the "corridor") and
an assignment of optimizer points to those polytopes, solves for the shortest
smooth path whose points each lie inside their assigned polytope.

The optimization problem is convex (an SOCP) with three components:

  Objective:
    1. Path length:   minimize sum of segment norms  (pulls the path tight,
                      cuts corners wherever the corridor allows)
    2. Smoothness:    minimize sum of squared discrete accelerations
                      (rounds off kinks at polytope boundaries)

  Constraints:
    1. Endpoint pins: P[0] == start, P[-1] == goal
    2. Corridor membership: each point stays in its assigned rectangle
       (four linear inequalities: x_min <= x <= x_max, y_min <= y <= y_max)
    3. Optional step-size cap: ||P[i+1] - P[i]|| <= d_max

Because the problem is convex, the CLARABEL solver finds the global optimum —
no local minima, no need for initial guess tuning.
"""

import numpy as np
import cvxpy as cp


def optimize_path_corridor(
    start: tuple[float, float],
    goal: tuple[float, float],
    corridor: list,                 # list of Polytope objects
    region_assignment: list[int],   # length n_points, each value in [0, len(corridor))
    smoothness_weight: float = 0.5,
    d_max: float | None = None,
    solver: str = 'CLARABEL',
    verbose: bool = False,
) -> np.ndarray:
    """Solve the corridor-constrained SOCP to find a short, smooth path.

    Parameters
    ----------
    start, goal : (x, y) tuples
        Exact start and goal positions. Must lie inside their assigned polytopes
        (corridor[region_assignment[0]] and corridor[region_assignment[-1]]).
    corridor : list of Polytope
        Sequence of obstacle-free rectangular regions from build_corridor().
        Each has .x_min, .x_max, .y_min, .y_max bounds.
    region_assignment : list of int, length n_points
        Entry i says which polytope optimizer point P[i] is constrained to.
        Produced by assign_points_to_regions().
    smoothness_weight : float
        Weight on the squared-acceleration smoothness term.
          0.0  = pure length minimization (shortest path, sharp kinks at
                 polytope boundaries)
          0.5  = balanced length + smoothness (default — rounds corners
                 without significantly increasing path length)
          large = very smooth but longer path
    d_max : float or None
        If set, caps the distance between consecutive optimizer points.
        Helps numerical conditioning and keeps point spacing roughly uniform.
        None = unconstrained.
    solver, verbose : passed to CVXPY's prob.solve().

    Returns
    -------
    (n_points, 2) numpy array of optimized (x, y) coordinates.
    """
    n = len(region_assignment)   # total number of optimizer points
    if n < 2:
        raise ValueError(f"Need at least 2 points; got {n}.")

    # Validate that start/goal lie in their assigned polytopes
    r_start = corridor[region_assignment[0]]
    r_goal  = corridor[region_assignment[-1]]
    if not r_start.contains(start):
        raise ValueError(f"Start point {start} not inside its assigned polytope.")
    if not r_goal.contains(goal):
        raise ValueError(f"Goal point {goal} not inside its assigned polytope.")

    # ---- Decision Variable ----
    # P is an (n x 2) matrix. P[i] = (x_i, y_i) is the position of the i-th
    # optimizer point. CVXPY will fill in the values that minimize the objective.
    P = cp.Variable((n, 2))

    # ---- Objective: Path Length ----
    # segments[i] = P[i+1] - P[i], the displacement vector of the i-th step.
    # cp.norm(segments, 2, axis=1) gives the Euclidean length of each step.
    # cp.sum(...) adds them up to give total path length.
    # Minimizing this pulls the path tight, like a string under tension.
    segments = P[1:] - P[:-1]
    length_term = cp.sum(cp.norm(segments, 2, axis=1))

    # ---- Objective: Smoothness (Acceleration) ----
    # The discrete second derivative (acceleration) at interior point i is:
    #   a_i = P[i-1] - 2*P[i] + P[i+1]
    # Minimizing sum of squared accelerations produces smooth, gradual turns
    # rather than sharp kinks where the path crosses from one polytope to the next.
    if n >= 3:
        accel = P[:-2] - 2 * P[1:-1] + P[2:]
        smoothness_term = cp.sum_squares(accel)
        objective = cp.Minimize(length_term + smoothness_weight * smoothness_term)
    else:
        # Fewer than 3 points: no interior points, so no smoothness to penalize
        objective = cp.Minimize(length_term)

    # ---- Constraints ----
    constraints = []

    # 1. Pin start and goal exactly
    constraints.append(P[0]  == np.asarray(start, dtype=float))
    constraints.append(P[-1] == np.asarray(goal,  dtype=float))

    # 2. Corridor membership: each point must lie inside its assigned polytope.
    #    For axis-aligned rectangles this is four linear inequalities per point,
    #    which CVXPY handles very efficiently (no cone needed, just LP constraints).
    for i in range(n):
        poly = corridor[region_assignment[i]]
        constraints.append(P[i, 0] >= poly.x_min)   # x >= x_min
        constraints.append(P[i, 0] <= poly.x_max)   # x <= x_max
        constraints.append(P[i, 1] >= poly.y_min)   # y >= y_min
        constraints.append(P[i, 1] <= poly.y_max)   # y <= y_max

    # 3. Optional step-size cap (second-order cone constraint)
    if d_max is not None:
        for i in range(n - 1):
            constraints.append(cp.norm(P[i + 1] - P[i], 2) <= d_max)

    # ---- Solve ----
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=verbose)

    if prob.status not in ('optimal', 'optimal_inaccurate'):
        raise ValueError(f"Solver returned status '{prob.status}'")

    return np.asarray(P.value)
