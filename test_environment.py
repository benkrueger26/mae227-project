from src.environment import GridEnvironment, RectObstacle
from src.visualize import EnvironmentVisualizer
from src.astar import astar_search, downsample_collinear

def main():

    #Initialize the environment
    env = GridEnvironment(width = 10, height = 8)
    
    # Add obstalces:
    #small square
    env.add_obstacle(RectObstacle(x_start = 2, y_start = 2, width = 2, height = 2))

    #tallthin rectangle
    env.add_obstacle(RectObstacle(x_start = 6, y_start = 1, width = 1, height =5))

    env.add_obstacle(RectObstacle(x_start = 1, y_start = 5, width = 4, height = 2))
    env.add_obstacle(RectObstacle(x_start = 8, y_start = 2, width = 2, height = 3))
    env.add_obstacle(RectObstacle(x_start = 0, y_start = 0, width = 3, height = 1))


    # Set start and goal in free space
    env.set_start(5,4)
    env.set_goal(9,1)

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
    viz.show()

    
if __name__ == "__main__":
    main()