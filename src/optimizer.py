import numpy as np
import cvxpy as cp

def optimize_path(q, bubbles, lambda_reg=1.0, d_max=1.5):
    """
    q: (K, 2) numpy array of reference waypoints
    bubbles: list of (cx, cy, r) tuples, length K
    lambda_reg: weight on the Tikhonov regularization term
    d_max: max allowed distance between consecutive optimized points
    
    Returns: (K, 2) numpy array of optimized waypoints
    """

    q = np.asarray(q, dtype = float)
    K = q.shape[0]

    #Ensure that every waypoint has 1 safetey bubble
    assert len(bubbles) == K, f"Mistmatch: {K} waypoints but {len(bubbles)} bubbles."

    #Define Decision variables... P is the K by 2 matrix of optimized (x,y) coordinates
    P = cp.Variable((K,2))

    # Define the objective function
    # Smoothness term (acceleration)
    # we want to minimize the difference between consecutive step vectors
    # Vectorized explanation:
    # P[:-2]   -> P[0] to P[K-3] (The "previous" points)
    # P[1:-1]  -> P[1] to P[K-2] (The "current" points)
    # P[2:]    -> P[2] to P[K-1] (The "next" points)
    # This perfectly calculates (P_{i-1} - 2P_i + P_{i+1}) for all interior points at once.
    accel_cost = cp.sum_squares(P[:-2] - 2 * P[1:-1] + P[2:])

    #Tikhonov term
    # We want to mimimze the squared distance between our optimized points P and the original A* path q so the path doesnt drift
    tikhonov_cost = cp.sum_squares(P-q)

    #Full objective function:
    objective = cp.Minimize(accel_cost + lambda_reg*tikhonov_cost)

    #Define Constraints
    constraints = []

    #Endpoint anchoring (start and end of optimized path must exactly match the start and end of the A* path)
    constraints += [
        P[0] == q[0],
        P[-1] == q[-1]
    ]

    #Every optimized point must stay inside the radius of it's corresponding bubble... extract center and radius from the bubbles
    for i in range(K):
        r = bubbles[i][2]
        constraints.append(cp.norm(P[i] - q[i], 2) <= r)

    #Disatnce between any point and the next one cannot exceed d_max to prevent bunching points at start and end
    for i in range(K-1):
        constraints.append(cp.norm(P[i+1] - P[i], 2) <= d_max)

    prob = cp.Problem(objective, constraints)

    prob.solve(solver=cp.CLARABEL)

    #Catch issues with the solver
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Solver failed to find an optimal solution. Status: {prob.status}")
    
    return P.value