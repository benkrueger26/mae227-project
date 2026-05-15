"""
environment.py — Grid world representation for motion planning.

Two classes:
  - RectObstacle: a single axis-aligned rectangular obstacle defined by its
    bottom-left corner, width, and height in continuous world coordinates.
  - GridEnvironment: the 2D map. Stores the grid size, obstacle list, start,
    and goal. Exposes an occupancy_grid property that rasterizes obstacles
    into a boolean numpy array for use by A* and the distance-field computation.

Coordinate convention:
  - World coordinates are continuous floats (e.g. 1.5, 7.3).
  - Cell (i, j) occupies the unit square from (i, j) to (i+1, j+1).
  - The center of cell (i, j) is at (i+0.5, j+0.5).
  - The occupancy grid is indexed as grid[row, col] = grid[y, x].
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class RectObstacle:
    """An axis-aligned rectangular obstacle.

    Parameters
    ----------
    x_start, y_start : float
        Bottom-left corner in world coordinates.
    width, height : float
        Extent in the x and y directions. Both must be positive.
    """
    x_start: float
    y_start: float
    width: float
    height: float

    def __post_init__(self):
        # Guard against zero- or negative-size obstacles, which would silently
        # block nothing and confuse the corridor builder's face-expansion logic.
        if self.width <= 0:
            raise ValueError(f"Obstacle width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"Obstacle height must be positive, got {self.height}")

    def contains(self, test_x: float, test_y: float) -> bool:
        """Return True if the point (test_x, test_y) is strictly inside this
        obstacle (half-open interval: includes left/bottom edge, excludes
        right/top edge — consistent with how occupancy_grid tests cell centers).
        """
        in_x_bounds = self.x_start <= test_x < (self.x_start + self.width)
        in_y_bounds = self.y_start <= test_y < (self.y_start + self.height)
        return in_x_bounds and in_y_bounds


class GridEnvironment:
    """A 2D grid-based planning environment.

    The grid is width x height cells. Obstacles are stored as a list of
    RectObstacle objects and rasterized on demand via the occupancy_grid
    property. Start and goal are continuous world coordinates (not cell indices).
    """

    def __init__(self, width: int, height: int):
        """Create an empty grid of the given size with no obstacles."""
        self.width = width
        self.height = height
        self.obstacles = []
        self.start = None   # set via set_start()
        self.goal = None    # set via set_goal()

    def add_obstacle(self, obstacle: RectObstacle) -> None:
        """Add a rectangular obstacle to the environment.

        Raises ValueError if the obstacle falls outside the grid bounds.
        """
        if obstacle.x_start < 0 or obstacle.y_start < 0:
            raise ValueError(
                f"Obstacle out of bounds: Starting Coordinates "
                f"({obstacle.x_start}, {obstacle.y_start}) cannot be negative"
            )
        if (obstacle.x_start + obstacle.width) > self.width:
            raise ValueError(
                f"Obstacle out of bounds: Extends past grid width of {self.width}"
            )
        if (obstacle.y_start + obstacle.height) > self.height:
            raise ValueError(
                f"Obstacle out of bounds: Extends past grid height of {self.height}"
            )
        self.obstacles.append(obstacle)

    def set_start(self, x: float, y: float) -> None:
        """Set the start position in continuous world coordinates.

        Raises ValueError if the position is out of bounds or inside an obstacle.
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"Start Coordinate ({x}, {y}) is out of grid bounds.")
        for obstacle in self.obstacles:
            if obstacle.contains(x, y):
                raise ValueError(f"Start Coordinate ({x}, {y}) is inside an obstacle.")
        self.start = (x, y)

    def set_goal(self, x: float, y: float) -> None:
        """Set the goal position in continuous world coordinates.

        Raises ValueError if the position is out of bounds or inside an obstacle.
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"Goal Coordinate ({x}, {y}) is out of grid bounds.")
        for obstacle in self.obstacles:
            if obstacle.contains(x, y):
                raise ValueError(f"Goal Coordinate ({x}, {y}) is inside an obstacle.")
        self.goal = (x, y)

    @property
    def occupancy_grid(self) -> np.ndarray:
        """Rasterize the obstacle list into a (height, width) boolean array.

        grid[y, x] is True if the center of cell (x, y) falls inside any obstacle.
        Testing the cell center (x+0.5, y+0.5) rather than the corner means a
        cell is only marked occupied when an obstacle actually covers its center —
        cells that share only a boundary with an obstacle remain free.

        Note: this is recomputed from scratch each call. Cache it locally if you
        need it many times (e.g. `df = compute_distance_field(env)` is one call).
        """
        grid = np.zeros((self.height, self.width), dtype=bool)
        for y in range(self.height):
            for x in range(self.width):
                for obstacle in self.obstacles:
                    if obstacle.contains(x + 0.5, y + 0.5):
                        grid[y, x] = True
                        break   # no need to check other obstacles for this cell
        return grid

    def is_free(self, x: int, y: int) -> bool:
        """Return True if integer cell (x, y) is inside the grid and not blocked.

        Used by A* to check whether a neighbor is a valid move target.
        """
        # Reject out-of-bounds cells
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        # Check against every obstacle using the cell-center convention
        for obstacle in self.obstacles:
            if obstacle.contains(x + 0.5, y + 0.5):
                return False
        return True
