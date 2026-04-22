import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from .environment import GridEnvironment


def compute_distance_field(env: GridEnvironment) -> np.ndarray:
    """Compute the euclidean distance from every free cell to every pbstacle"""

    #env.occupancy_grid is true for obstacles and false for free space
    #distance_transform_edt  calculates the distance to the nearest zero
    # we invert the grid to make the obstacles have value zero and free space have value 1

    free_space = ~env.occupancy_grid
    return distance_transform_edt(free_space)

def compute_bubbles(
    waypoints: list[tuple[float, float]],
    env: GridEnvironment,
    safety_margin: float = 0.2,
    min_radius: float = 0.0,
) -> list[tuple[float, float, float]]:
    """Create safe circular bubbles around each waypoint using exact
    distance to axis-aligned rectangular obstacles."""
    bubbles = []
    for wx, wy in waypoints:
        min_dist = float('inf')
        for obs in env.obstacles:
            # Closest point on the rectangle to (wx, wy)
            cx = max(obs.x_start, min(wx, obs.x_start + obs.width))
            cy = max(obs.y_start, min(wy, obs.y_start + obs.height))
            d = np.hypot(wx - cx, wy - cy)
            min_dist = min(min_dist, d)

        # Also respect grid boundaries
        min_dist = min(min_dist, wx, wy,
                       env.width - wx, env.height - wy)

        r = max(min_dist - safety_margin, min_radius)
        bubbles.append((wx, wy, r))
    return bubbles