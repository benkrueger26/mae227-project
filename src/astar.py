"""
astar.py — A* pathfinding on a 2D grid.

Exposes three functions used by the planning pipeline:

  astar_search         — finds the shortest discrete path from start to goal.
  downsample_collinear — strips redundant waypoints from straight runs,
                         keeping only the corners where direction changes.
  resample_by_spacing  — inserts interpolated points so no two consecutive
                         waypoints are farther apart than a given distance,
                         giving the SOCP optimizer enough points to represent
                         smooth curves.
"""

import heapq
import math
from typing import Optional
from .environment import GridEnvironment


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

def euclidean_heuristic(state: tuple[int, int], goal: tuple[int, int]) -> float:
    """Straight-line (Euclidean) distance from state to goal.

    Admissible for 8-connected grids: the true cost can never be less than
    the straight-line distance, so this never over-estimates — A* stays optimal.
    """
    return math.hypot(goal[0] - state[0], goal[1] - state[1])


# ---------------------------------------------------------------------------
# Path reconstruction helper
# ---------------------------------------------------------------------------

def _reconstruct_path(came_from: dict, goal: tuple[int, int]) -> list[tuple[int, int]]:
    """Walk the backpointer chain from goal back to start, then reverse.

    came_from[node] stores which cell we came from to reach `node`.
    came_from[start] is None, which terminates the loop.
    After reversal, the list runs from start to goal.
    """
    path = [goal]
    while came_from[path[-1]] is not None:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Core A* search
# ---------------------------------------------------------------------------

def astar_search(
    env: GridEnvironment,
    heuristic=euclidean_heuristic,
    disallow_corner_cutting: bool = False,
) -> Optional[list[tuple[int, int]]]:
    """Run A* from env.start to env.goal on an 8-connected grid.

    A* is a best-first search that always expands the cell with the lowest
    estimated total cost f = g + h, where:
      g = actual cost to reach this cell from the start (path length so far)
      h = heuristic estimate of cost from here to the goal (Euclidean distance)

    Because h is admissible (never over-estimates), A* is guaranteed to find
    the optimal (shortest) path if one exists.

    Parameters
    ----------
    env : GridEnvironment
        The map. Uses env.is_free() to check if a cell is traversable.
    heuristic : callable
        Function (state, goal) -> float. Default: Euclidean distance.
    disallow_corner_cutting : bool
        If True, diagonal moves are blocked when either adjacent cardinal
        neighbor is an obstacle. This prevents paths from squeezing through
        a one-cell diagonal gap — important when the corridor builder checks
        continuous segment geometry against inflated obstacles.

    Returns
    -------
    List of (x, y) integer cell tuples from start to goal, or None if
    no path exists (the environment is completely blocked).
    """
    if env.start is None or env.goal is None:
        return None

    # Convert continuous start/goal coordinates to integer cell indices.
    # int() truncates: (1.5, 2.7) → cell (1, 2), the cell containing that point.
    start = (int(env.start[0]), int(env.start[1]))
    goal  = (int(env.goal[0]),  int(env.goal[1]))

    # All 8 neighbours: 4 cardinal (cost 1) + 4 diagonal (cost sqrt(2))
    moves = [
        (1, 0), (-1, 0), (0, 1), (0, -1),   # right, left, up, down
        (1, 1), (1, -1), (-1, 1), (-1, -1)   # diagonals
    ]

    # Priority queue ordered by f = g + h.
    # Python's heapq is a min-heap, so the lowest-f cell is always popped first.
    frontier = []
    heapq.heappush(frontier, (heuristic(start, goal), start))

    # g_score[s] = cost of the cheapest known path from start to cell s
    g_score = {start: 0.0}

    # came_from[s] = which cell we came from to reach s (the backpointer).
    # The start node has no parent, so it maps to None.
    came_from = {start: None}

    while frontier:
        f_current, current = heapq.heappop(frontier)

        # If we reached the goal, reconstruct and return the full path
        if current == goal:
            return _reconstruct_path(came_from, goal)

        for dx, dy in moves:
            neighbor = (current[0] + dx, current[1] + dy)

            # Skip cells outside the grid or inside obstacles
            if not env.is_free(neighbor[0], neighbor[1]):
                continue

            # Corner-cutting check: for a diagonal move (both dx and dy nonzero),
            # require both axis-aligned cardinal neighbors to also be free.
            # Without this, a diagonal step can pass through the corner shared
            # by two diagonally adjacent obstacles, which is geometrically invalid
            # for a finite-radius robot body.
            if disallow_corner_cutting and dx != 0 and dy != 0:
                if not env.is_free(current[0] + dx, current[1]):
                    continue
                if not env.is_free(current[0], current[1] + dy):
                    continue

            # Step cost: 1.0 for cardinal, sqrt(2) ≈ 1.414 for diagonals
            step_cost = math.hypot(dx, dy)
            tentative_g = g_score[current] + step_cost

            # Only update if this is a cheaper route to the neighbor than
            # any previously found. If g_score[neighbor] doesn't exist yet,
            # .get() returns inf, so the first visit always wins.
            if tentative_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f_neighbor = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(frontier, (f_neighbor, neighbor))

    # Frontier exhausted without reaching goal → no path exists
    return None


# ---------------------------------------------------------------------------
# Waypoint post-processing
# ---------------------------------------------------------------------------

def downsample_collinear(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Strip interior waypoints where the direction of travel hasn't changed.

    A* on a grid produces one cell per step. On a straight diagonal run of
    10 cells, that's 10 waypoints — but only the start and end of that run
    carry topological information. This function keeps only the "corners":
    points where the direction vector changes.

    Example: a path going right for 5 then up for 3 goes from ~8 waypoints
    down to 3 (start, the corner, goal).

    Input : dense A* path (list of integer (x, y) cell tuples)
    Output: corner-only path (same start/end, fewer interior points)
    """
    if not path or len(path) <= 2:
        return path[:]

    result = [path[0]]  # always keep the start

    for i in range(1, len(path) - 1):
        prev = path[i - 1]
        curr = path[i]
        nxt  = path[i + 1]

        # Direction from previous cell to this one
        dir_in  = (curr[0] - prev[0], curr[1] - prev[1])
        # Direction from this cell to the next one
        dir_out = (nxt[0]  - curr[0], nxt[1]  - curr[1])

        # If the direction changed, this is a corner — keep it
        if dir_in != dir_out:
            result.append(curr)

    result.append(path[-1])  # always keep the goal
    return result


def resample_by_spacing(
    waypoints: list[tuple[float, float]],
    max_spacing: float = 1.5,
) -> list[tuple[float, float]]:
    """Insert linearly interpolated points so no two adjacent waypoints are
    more than max_spacing apart in Euclidean distance.

    Why this is needed: after downsample_collinear, long straight segments
    may have no interior points. The SOCP optimizer and bubble computations
    each need a point within the segment to place a bubble / assign a
    variable. Without interior points, the optimizer has no freedom to curve
    a long segment — it can only move the endpoints.

    All original input waypoints (corners) are always preserved exactly.
    New points are evenly spaced by linear interpolation between each pair.

    Parameters
    ----------
    waypoints   : list of (x, y) floats — the sparse corner-only path
    max_spacing : maximum allowed distance between consecutive output points
    """
    if len(waypoints) < 2:
        return waypoints[:]

    result = [waypoints[0]]

    for i in range(len(waypoints) - 1):
        p0 = waypoints[i]
        p1 = waypoints[i + 1]

        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        segment_length = math.hypot(dx, dy)

        # Divide the segment into enough equal pieces so each is <= max_spacing
        n_segments = max(1, math.ceil(segment_length / max_spacing))

        # Insert interior points at t = 1/n, 2/n, ..., (n-1)/n, then the endpoint
        for k in range(1, n_segments):
            t = k / n_segments          # fraction along segment, in (0, 1)
            interp = (p0[0] + t * dx, p0[1] + t * dy)
            result.append(interp)
        result.append(p1)               # always include the original corner

    return result
