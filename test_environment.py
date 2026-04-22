from src.environment import GridEnvironment, RectObstacle
from src.visualize import EnvironmentVisualizer
from src.astar import astar_search, downsample_collinear, resample_by_spacing
from src.bubbles import compute_distance_field, compute_bubbles


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
       waypoints_dense = resample_by_spacing(waypoints, max_spacing=1.5)
       print(f"After resampling: {len(waypoints_dense)} waypoints")    

    # Shift waypoints from cell-index coordinates to world (cell-center) coordinates.
    # This aligns with the visualizer's +0.5 render convention and with the
    # distance field's cell-center distance measurements.
    waypoints_world = [(x + 0.5, y + 0.5) for x, y in waypoints_dense]

    # Generate the distance field and bubbles
    df = compute_distance_field(env)
    bubbles = compute_bubbles(waypoints_world, env)
    
    

    print("\n--- Safety Bubbles ---")
    for i, (cx, cy, r) in enumerate(bubbles):
        print(f"Waypoint {i} at ({cx}, {cy}) -> Radius: {r:.2f}")

    # Diagnostic: check a specific waypoint against the grid
    print("\n--- Bubble Diagnostic ---")
    print(f"Distance field shape: {df.shape}")
    print(f"Grid shape: {env.occupancy_grid.shape}")

    # Check the first few waypoints
    for i, ((wx, wy), (bx, by, br)) in enumerate(zip(waypoints_world, bubbles)):
        # Sample the raw distance field at this waypoint
        from scipy.ndimage import map_coordinates
        import numpy as np
        raw_dist = map_coordinates(df, np.array([[wy], [wx]]), order=1, mode='nearest')[0]
        
        # Find the nearest obstacle cell by brute force
        obs_ys, obs_xs = np.where(env.occupancy_grid)
        if len(obs_xs) > 0:
            # Distance from waypoint to obstacle cell CENTERS
            dists_to_centers = np.sqrt((obs_xs + 0.5 - wx)**2 + (obs_ys + 0.5 - wy)**2)
            # Distance from waypoint to obstacle cell EDGES (closest point on cell square)
            dx = np.maximum(0, np.abs(wx - (obs_xs + 0.5)) - 0.5)
            dy = np.maximum(0, np.abs(wy - (obs_ys + 0.5)) - 0.5)
            dists_to_edges = np.sqrt(dx**2 + dy**2)
            
            print(f"WP {i}: world=({wx:.2f},{wy:.2f}) "
                f"df_sample={raw_dist:.3f} "
                f"true_center_dist={dists_to_centers.min():.3f} "
                f"true_edge_dist={dists_to_edges.min():.3f} "
                f"bubble_r={br:.3f}")
        
        if i >= 5:
            break

    viz = EnvironmentVisualizer(env)

    if path:
        viz.draw_astar_path(path)
        viz.draw_waypoints(waypoints_dense)
        viz.draw_bubbles(bubbles)

    viz.show()


    
if __name__ == "__main__":
    main()