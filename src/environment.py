from dataclasses import dataclass
import numpy as np

@dataclass
class RectObstacle:
    x_start:float
    y_start: float
    width: float
    height: float

    def contains(self, test_x: float, test_y: float) -> bool:
        """Check whether a given grid coordinate falls inside this obstacle"""
        in_x_bounds = self.x_start <= test_x < (self.x_start + self.width)
        in_y_bounds = self.y_start <= test_y < (self.y_start + self.height)
        return in_x_bounds and in_y_bounds
    
    def __post_init__(self):
        if self.width <= 0:
            raise ValueError(f"Obstacle width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"Obstacle height must be positive, got {self.height}")
    

class GridEnvironment:
    def __init__(self, width: int, height: int):
        """Create an N x M grid with no obstacles, no start, no goal."""
        self.width = width
        self.height = height
        self.obstacles = []
        self.start = None # No start set yet
        self.goal = None # No goal set yet
    
    def add_obstacle(self, obstacle: RectObstacle) -> None:
        """Append an obstacle to the environment."""

        if obstacle.x_start < 0 or obstacle.y_start < 0:
            raise ValueError(f"Obstacle out of bounds: Starting Coordinates ({obstacle.x_start}, {obstacle.y_start}) cannot be negative")
        
        if (obstacle.x_start + obstacle.width) > self.width:
            raise ValueError(f"Obstacle out of bounds: Extends past grid width of {self.width}")
        
        if (obstacle.y_start + obstacle.height) > self.height:
            raise ValueError(f"Obstacle out of bounds: Extends past grid height of {self.height}")
        
        #Add obstacle:
        self.obstacles.append(obstacle)

    
    def set_start(self, x: float, y: float) -> None:
        # Check if starting coord is in grid
        if not (0<= x < self.width and 0<= y < self.height):
            raise ValueError(f"Start Coordinate ({x}, {y}) is out of grid bounds. ")
        
        for obstacle in self.obstacles:
            if obstacle.contains(x,y):
                raise ValueError(f"Start Coordinate ({x}, {y}) is inside an obstacle. ")
            
        # if the coordinate is valid, set as start position
        self.start = (x,y)
    
    def set_goal(self, x: float, y: float) -> None:
        # Check if goal coord is in grid
        if not (0<= x < self.width and 0<= y < self.height):
            raise ValueError(f"Goal Coordinate ({x}, {y}) is out of grid bounds. ")
        
        for obstacle in self.obstacles:
            if obstacle.contains(x,y):
                raise ValueError(f"Goal Coordinate ({x}, {y}) is inside an obstacle. ")
            
        # if the coordinate is valid, set as goal position
        self.goal = (x,y)
    
    @property
    def occupancy_grid(self) -> np.ndarray:
        """
        Return a (height, width) boolean array where True = blocked.
        Rasterized from the current obstacle list.
        """
        #Initizalize array
        grid = np.zeros((self.height, self.width), dtype = bool)

        for y in range(self.height):
            for x in range(self.width):
                for obstacle in self.obstacles:
                    if obstacle.contains(x+0.5, y+0.5): #Cell is blocked
                        grid[y,x] = True
                        break

        return grid
    
    def is_free(self, x: int, y: int) -> bool:
        """True if cell (x, y) is inside the grid and not blocked."""

        #Check if inbounds
        if not (0 <= x <self.width and 0 <=y< self.height):
            return False
        
        #Check inside obstacles
        for obstacle in self.obstacles:
            if obstacle.contains(x+0.5, y+0.5):
                return False
        
        #Unblocked and not inside obstacle, it is free
        return True
        

