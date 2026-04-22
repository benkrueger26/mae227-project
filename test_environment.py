from src.environment import GridEnvironment, RectObstacle
from src.visualize import EnvironmentVisualizer
from src.astar import astar_search, downsample_collinear

def main():

    #Initialize the environment
    env = GridEnvironment(width = 20, height = 20)
    
    # Add obstalces:
    #small square
    # env.add_obstacle(RectObstacle(x_start = 2, y_start = 2, width = 2, height = 2))

    # #tallthin rectangle
    # env.add_obstacle(RectObstacle(x_start = 6, y_start = 1, width = 1, height =5))

    # env.add_obstacle(RectObstacle(x_start = 1, y_start = 5, width = 4, height = 2))
    # env.add_obstacle(RectObstacle(x_start = 18, y_start = 2, width = 2, height = 6))
    # env.add_obstacle(RectObstacle(x_start = 0, y_start = 10, width = 3, height = 1))
    # env.add_obstacle(RectObstacle(x_start = 10, y_start = 9, width = 5, height = 7))

    env = GridEnvironment(width=20, height=20)

    env = GridEnvironment(width=20, height=20)

    # Bottom-left: a zigzag corridor
    env.add_obstacle(RectObstacle(x_start=2,  y_start=0,  width=2, height=5))
    env.add_obstacle(RectObstacle(x_start=5,  y_start=3,  width=2, height=5))

    # Middle: horizontal wall with a narrow gap, forces a specific crossing
    env.add_obstacle(RectObstacle(x_start=0,  y_start=9,  width=8, height=2))
    env.add_obstacle(RectObstacle(x_start=10, y_start=9,  width=10, height=2))
    # Gap is between x=8 and x=10

    # Upper-middle: a diagonal chain of pillars that forces sinuous motion
    env.add_obstacle(RectObstacle(x_start=3,  y_start=12, width=2, height=2))
    env.add_obstacle(RectObstacle(x_start=7,  y_start=14, width=2, height=2))
    env.add_obstacle(RectObstacle(x_start=11, y_start=16, width=2, height=2))

    # Upper-right: a hook that the path has to curve around
    env.add_obstacle(RectObstacle(x_start=15, y_start=13, width=4, height=2))
    env.add_obstacle(RectObstacle(x_start=15, y_start=15, width=2, height=4))

    # Right side below the wall: a block forcing the path to stay left
    env.add_obstacle(RectObstacle(x_start=14, y_start=3,  width=3, height=4))

    
    env.set_start(0.5, 0.5)
    env.set_goal(19.5, 19.5)



    # Set start and goal in free space
    """x, y are continuous world coordinates. To target the center of cell (i, j), pass (i+0.5, j+0.5)."""
    env.set_start(0+0.5,1+0.5)
    env.set_goal(19+0.5,17+0.5)

    #Print grid as ascii
    grid = env.occupancy_grid
    start_x,start_y = int(env.start[0]), int(env.start[1])
    goal_x,goal_y = int(env.goal[0]), int(env.goal[1])
    
    print("--- Grid Environment ASCII Test ---")
    for y in range(env.height -1, -1, -1):
        row_chars = []
        for x in range(env.width):
            if (x,y) == (start_x, start_y):
                row_chars.append('S')
            elif (x,y) == (goal_x, goal_y):
                row_chars.append('G')
            elif grid[y,x]:
                row_chars.append('#')
            else:
                row_chars.append('.')
        print(' '.join(row_chars))


   
    path = astar_search(env)
    if path is None:
        print("No path found!")
    else:
       print(f"A* path: {len(path)} waypoints")
       waypoints = downsample_collinear(path)
       print(f"After downsampling: {len(waypoints)} waypoints")
       print(f"Waypoints: {waypoints}")

    viz = EnvironmentVisualizer(env)

    if path:
        viz.draw_astar_path(path)
        viz.draw_waypoints(waypoints)
    viz.show()

    
if __name__ == "__main__":
    main()