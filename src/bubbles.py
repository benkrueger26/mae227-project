"""
bubbles.py — Distance field computation and safety bubble generation.

Used by the bubble-based SOCP pipeline (test_environment.py).

A "safety bubble" is a circle centered on a waypoint whose radius is the
distance from that waypoint to the nearest obstacle surface. Any point
inside the bubble is guaranteed to be in free space. The SOCP optimizer
uses these bubbles as constraints: each optimized point must stay within
its bubble, which guarantees the optimized path is collision-free.

Two functions:
  compute_distance_field — builds a scalar field over the whole grid
                           giving the Euclidean distance to the nearest
                           obstacle at every point.
  compute_bubbles        — computes one (cx, cy, radius) bubble per waypoint
                           using exact geometry against the obstacle list.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from .environment import GridEnvironment


def compute_distance_field(env: GridEnvironment) -> np.ndarray:
    """Compute the Euclidean distance-to-obstacle field for the whole grid.

    Returns a (height, width) float array where grid[y, x] is the distance
    from cell (x, y) to the nearest obstacle cell.

    How it works:
      scipy's distance_transform_edt computes, for every True cell in a binary
      array, the distance to the nearest False cell. We pass it the INVERTED
      occupancy grid (free = True, obstacle = False) so the result measures
      distance from each free cell to the nearest obstacle.
    """
    # occupancy_grid: True = obstacle, False = free.
    # Invert so free cells are 1 and obstacles are 0;
    # EDT then gives distance from each free cell to the nearest 0 (obstacle).
    free_space = ~env.occupancy_grid
    return distance_transform_edt(free_space)


def compute_bubbles(
    waypoints: list[tuple[float, float]],
    env: GridEnvironment,
    safety_margin: float = 0.2,
    min_radius: float = 0.0,
) -> list[tuple[float, float, float]]:
    """Compute one safe circular bubble per waypoint using exact geometry.

    For each waypoint (wx, wy), the bubble radius is the distance to the
    nearest obstacle surface minus a safety margin. Any point inside the
    bubble is guaranteed to be at least safety_margin away from every obstacle.

    The SOCP optimizer constrains each optimized point to stay within its
    bubble, which guarantees the final path is collision-free.

    Distance computation:
      For each axis-aligned rectangular obstacle, the closest point on the
      rectangle to (wx, wy) is found by clamping wx and wy to the rectangle's
      extents. The Euclidean distance to that clamped point is the exact
      distance from (wx, wy) to the nearest face of the obstacle.
      Grid boundaries are also treated as walls.

    Parameters
    ----------
    waypoints     : list of (x, y) world-coordinate floats
    env           : GridEnvironment with the obstacle list and grid bounds
    safety_margin : shrink each bubble by this amount for a clearance buffer
    min_radius    : floor on the bubble radius — prevents zero-radius bubbles
                    from completely locking the optimizer in place

    Returns
    -------
    List of (cx, cy, r) tuples — one per waypoint.
    """
    bubbles = []

    for wx, wy in waypoints:
        min_dist = float('inf')

        # Find the distance from (wx, wy) to the nearest surface of each obstacle
        for obs in env.obstacles:
            # Clamp (wx, wy) to the rectangle's extents to get the closest point
            # on the rectangle's surface. If (wx, wy) is inside the rectangle,
            # both clamps are no-ops and d = 0 (shouldn't happen for valid waypoints).
            cx = max(obs.x_start, min(wx, obs.x_start + obs.width))
            cy = max(obs.y_start, min(wy, obs.y_start + obs.height))
            d = np.hypot(wx - cx, wy - cy)
            min_dist = min(min_dist, d)

        # Also treat the four grid boundary walls as obstacles.
        # wx = distance to left wall, wy = distance to bottom wall,
        # width-wx = distance to right wall, height-wy = distance to top wall.
        min_dist = min(min_dist, wx, wy, env.width - wx, env.height - wy)

        # Subtract the safety margin; clamp to min_radius so we never go negative
        r = max(min_dist - safety_margin, min_radius)
        bubbles.append((wx, wy, r))

    return bubbles
