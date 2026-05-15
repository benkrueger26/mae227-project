"""
test_environment.py — Bubble-based SOCP motion planning pipeline (main demo).

Full pipeline:
  1. Build a 20x20 grid environment with obstacles, start (bottom-left),
     and goal (upper-right).
  2. Run A* (8-connected) to find a discrete shortest path.
  3. Downsample collinear cells to keep only direction-change corners.
  4. Resample at two densities:
       sparse (max_spacing=1.5) — for clean visualization (fewer plotted dots)
       dense  (max_spacing=0.25) — for the SOCP (more variables → smoother curve)
  5. Convert integer cell indices to world coordinates (+0.5 for cell centers).
  6. Compute a safety bubble for each dense waypoint: radius = distance to
     nearest obstacle surface minus safety_margin.
  7. Run the SOCP optimizer with lambda_reg=0 (pure rubber-band smoothing):
     find the smoothest polyline whose points each stay inside their bubble.
  8. Compare path lengths and visualize all layers with toggleable checkboxes.

Key tuning knobs:
  max_spacing in resample_by_spacing — controls optimizer point density
  lambda_reg  in optimize_path       — 0 = pure smoothing, high = close to A*
  d_max       in optimize_path       — step-length cap to prevent point bunching
  safety_margin in compute_bubbles   — clearance buffer between bubble and obstacle
"""

from src.environment import GridEnvironment, RectObstacle
from src.visualize import EnvironmentVisualizer
from src.astar import astar_search, downsample_collinear, resample_by_spacing
from src.bubbles import compute_distance_field, compute_bubbles
from src.optimizer import optimize_path
import numpy as np
from scipy.ndimage import map_coordinates


# def main():

#     #Initialize the environment
#     env = GridEnvironment(width = 20, height = 20)
    
#     # Add obstalces:
#     #small square
#     # env.add_obstacle(RectObstacle(x_start = 2, y_start = 2, width = 2, height = 2))

#     # #tallthin rectangle
#     # env.add_obstacle(RectObstacle(x_start = 6, y_start = 1, width = 1, height =5))

#     # env.add_obstacle(RectObstacle(x_start = 1, y_start = 5, width = 4, height = 2))
#     # env.add_obstacle(RectObstacle(x_start = 18, y_start = 2, width = 2, height = 6))
#     # env.add_obstacle(RectObstacle(x_start = 0, y_start = 10, width = 3, height = 1))
#     # env.add_obstacle(RectObstacle(x_start = 10, y_start = 9, width = 5, height = 7))

#     env = GridEnvironment(width=20, height=20)

#     env = GridEnvironment(width=20, height=20)

#     # Bottom-left: a zigzag corridor
#     env.add_obstacle(RectObstacle(x_start=2,  y_start=0,  width=2, height=5))
#     env.add_obstacle(RectObstacle(x_start=5,  y_start=3,  width=2, height=5))

#     # Middle: horizontal wall with a narrow gap, forces a specific crossing
#     env.add_obstacle(RectObstacle(x_start=0,  y_start=9,  width=8, height=2))
#     env.add_obstacle(RectObstacle(x_start=10, y_start=9,  width=10, height=2))
#     # Gap is between x=8 and x=10

#     # Upper-middle: a diagonal chain of pillars that forces sinuous motion
#     env.add_obstacle(RectObstacle(x_start=3,  y_start=12, width=2, height=2))
#     env.add_obstacle(RectObstacle(x_start=7,  y_start=14, width=2, height=2))
#     env.add_obstacle(RectObstacle(x_start=11, y_start=16, width=2, height=2))

#     # Upper-right: a hook that the path has to curve around
#     env.add_obstacle(RectObstacle(x_start=15, y_start=13, width=4, height=2))
#     env.add_obstacle(RectObstacle(x_start=15, y_start=15, width=2, height=4))

#     # Right side below the wall: a block forcing the path to stay left
#     env.add_obstacle(RectObstacle(x_start=14, y_start=3,  width=3, height=4))

    
#     env.set_start(0.5, 0.5)
#     env.set_goal(19.5, 19.5)



#     # Set start and goal in free space
#     """x, y are continuous world coordinates. To target the center of cell (i, j), pass (i+0.5, j+0.5)."""
#     env.set_start(0+0.5,1+0.5)
#     env.set_goal(19+0.5,17+0.5)

#     #Print grid as ascii
#     grid = env.occupancy_grid
#     start_x,start_y = int(env.start[0]), int(env.start[1])
#     goal_x,goal_y = int(env.goal[0]), int(env.goal[1])
    
#     print("--- Grid Environment ASCII Test ---")
#     for y in range(env.height -1, -1, -1):
#         row_chars = []
#         for x in range(env.width):
#             if (x,y) == (start_x, start_y):
#                 row_chars.append('S')
#             elif (x,y) == (goal_x, goal_y):
#                 row_chars.append('G')
#             elif grid[y,x]:
#                 row_chars.append('#')
#             else:
#                 row_chars.append('.')
#         print(' '.join(row_chars))


   
#     path = astar_search(env)
#     if path is None:
#         print("No path found!")
#     else:
#        print(f"A* path: {len(path)} waypoints")
#        waypoints = downsample_collinear(path)
#        print(f"After downsampling: {len(waypoints)} waypoints")
#        print(f"Waypoints: {waypoints}")
#        waypoints_dense = resample_by_spacing(waypoints, max_spacing=0.75)
#        print(f"After resampling: {len(waypoints_dense)} waypoints")    

#     # Shift waypoints from cell-index coordinates to world (cell-center) coordinates.
#     # This aligns with the visualizer's +0.5 render convention and with the
#     # distance field's cell-center distance measurements.
#     waypoints_world = [(x + 0.5, y + 0.5) for x, y in waypoints_dense]

#     # Generate the distance field and bubbles
#     df = compute_distance_field(env)
#     bubbles = compute_bubbles(waypoints_world, env)
    
#     q = np.asarray(waypoints_world, dtype=float)
#     P_opt = optimize_path(q, bubbles, lambda_reg=0.01, d_max=1.5)

#     print("\n--- Safety Bubbles ---")
#     for i, (cx, cy, r) in enumerate(bubbles):
#         print(f"Waypoint {i} at ({cx}, {cy}) -> Radius: {r:.2f}")

#     # Diagnostic: check a specific waypoint against the grid
#     print("\n--- Bubble Diagnostic ---")
#     print(f"Distance field shape: {df.shape}")
#     print(f"Grid shape: {env.occupancy_grid.shape}")

#     # Check the first few waypoints
#     for i, ((wx, wy), (bx, by, br)) in enumerate(zip(waypoints_world, bubbles)):
#         # Sample the raw distance field at this waypoint
        
#         raw_dist = map_coordinates(df, np.array([[wy], [wx]]), order=1, mode='nearest')[0]
        
#         # Find the nearest obstacle cell by brute force
#         obs_ys, obs_xs = np.where(env.occupancy_grid)
#         if len(obs_xs) > 0:
#             # Distance from waypoint to obstacle cell CENTERS
#             dists_to_centers = np.sqrt((obs_xs + 0.5 - wx)**2 + (obs_ys + 0.5 - wy)**2)
#             # Distance from waypoint to obstacle cell EDGES (closest point on cell square)
#             dx = np.maximum(0, np.abs(wx - (obs_xs + 0.5)) - 0.5)
#             dy = np.maximum(0, np.abs(wy - (obs_ys + 0.5)) - 0.5)
#             dists_to_edges = np.sqrt(dx**2 + dy**2)
            
#             print(f"WP {i}: world=({wx:.2f},{wy:.2f}) "
#                 f"df_sample={raw_dist:.3f} "
#                 f"true_center_dist={dists_to_centers.min():.3f} "
#                 f"true_edge_dist={dists_to_edges.min():.3f} "
#                 f"bubble_r={br:.3f}")
        
#         if i >= 5:
#             break

#     viz = EnvironmentVisualizer(env)

#     if path:
#         viz.draw_astar_path(path)
#         viz.draw_waypoints(waypoints_dense)
#         viz.draw_bubbles(bubbles)
#         viz.draw_optimized_path(P_opt)
#     viz.show()


    
# if __name__ == "__main__":
#     main()

def calculate_path_length(path_points):
    """Calculates the total Euclidean distance of a sequence of 2D points."""
    if path_points is None or len(path_points) < 2:
        return 0.0
    
    # Convert to numpy array (handles both lists of tuples and existing ndarrays)
    pts = np.asarray(path_points, dtype=float)
    
    # Calculate the vector differences between consecutive points
    diffs = np.diff(pts, axis=0)
    
    # Calculate the Euclidean length (norm) of each vector segment and sum them up
    return np.sum(np.linalg.norm(diffs, axis=1))

def main():
    # Initialize the environment
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

    # Set start and goal in free space
    # x, y are continuous world coordinates. To target the center of cell (i, j), pass (i+0.5, j+0.5).
    env.set_start(0+0.5, 1+0.5)
    env.set_goal(19+0.5, 17+0.5)

    # Print grid as ascii
    grid = env.occupancy_grid
    start_x, start_y = int(env.start[0]), int(env.start[1])
    goal_x, goal_y = int(env.goal[0]), int(env.goal[1])
    
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
        return  # Exit early if no path

    print(f"A* path: {len(path)} waypoints")
    waypoints = downsample_collinear(path)
    print(f"After downsampling: {len(waypoints)} waypoints")

    # --- BRANCHING FOR VISUALIZATION VS OPTIMIZATION ---

    # 1. SPARSE waypoints (for clean visualization)
    waypoints_sparse = resample_by_spacing(waypoints, max_spacing=1.5)
    print(f"Sparse waypoints for viz: {len(waypoints_sparse)}")

    # 2. DENSE waypoints (for continuous safety tube and smooth optimization)
    waypoints_dense = resample_by_spacing(waypoints, max_spacing=0.25)
    print(f"Dense waypoints for math: {len(waypoints_dense)}")

    # Shift both sets of waypoints to world (cell-center) coordinates
    waypoints_world_sparse = [(x + 0.5, y + 0.5) for x, y in waypoints_sparse]
    waypoints_world_dense = [(x + 0.5, y + 0.5) for x, y in waypoints_dense]

    # Generate the distance field 
    df = compute_distance_field(env)
    
    # Calculate bubbles for both sets
    bubbles_sparse = compute_bubbles(waypoints_world_sparse, env)
    bubbles_dense = compute_bubbles(waypoints_world_dense, env)

    # --- OPTIMIZE USING ONLY DENSE DATA ---
    q_dense = np.asarray(waypoints_world_dense, dtype=float)
    
    # lambda_reg=0.0 allows the path to pull tight like a rubber band!
    P_opt = optimize_path(q_dense, bubbles_dense, lambda_reg=0.0, d_max=1.5)

    print("\n--- Safety Bubbles (Dense) ---")
    for i, (cx, cy, r) in enumerate(bubbles_dense):
        if i < 5: # Only print first few to avoid console spam
            print(f"Dense Waypoint {i} at ({cx:.2f}, {cy:.2f}) -> Radius: {r:.2f}")

    # Diagnostic: check a specific dense waypoint against the grid
    print("\n--- Bubble Diagnostic ---")
    print(f"Distance field shape: {df.shape}")
    print(f"Grid shape: {env.occupancy_grid.shape}")

    for i, ((wx, wy), (bx, by, br)) in enumerate(zip(waypoints_world_dense, bubbles_dense)):
        raw_dist = map_coordinates(df, np.array([[wy], [wx]]), order=1, mode='nearest')[0]
        
        obs_ys, obs_xs = np.where(env.occupancy_grid)
        if len(obs_xs) > 0:
            dists_to_centers = np.sqrt((obs_xs + 0.5 - wx)**2 + (obs_ys + 0.5 - wy)**2)
            dx = np.maximum(0, np.abs(wx - (obs_xs + 0.5)) - 0.5)
            dy = np.maximum(0, np.abs(wy - (obs_ys + 0.5)) - 0.5)
            dists_to_edges = np.sqrt(dx**2 + dy**2)
            
            print(f"Dense WP {i}: world=({wx:.2f},{wy:.2f}) "
                  f"df_sample={raw_dist:.3f} "
                  f"true_center_dist={dists_to_centers.min():.3f} "
                  f"true_edge_dist={dists_to_edges.min():.3f} "
                  f"bubble_r={br:.3f}")
        
        if i >= 4:
            break

    # --- OPTIMIZE USING ONLY DENSE DATA ---
    q_dense = np.asarray(waypoints_world_dense, dtype=float)
    
    # lambda_reg=0.0 allows the path to pull tight like a rubber band!
    P_opt = optimize_path(q_dense, bubbles_dense, lambda_reg=0.0, d_max=1.5)

    # --- CALCULATE AND PRINT PATH LENGTHS ---
    astar_length = calculate_path_length(path)
    opt_length = calculate_path_length(P_opt)
    
    print("\n--- Path Length Comparison ---")
    print(f"Original A* Length: {astar_length:.2f} units")
    print(f"Optimized Length:   {opt_length:.2f} units")
    print(f"Distance Saved:     {astar_length - opt_length:.2f} units")

    # --- VISUALIZATION ---
    viz = EnvironmentVisualizer(env)

    if path:
        viz.draw_astar_path(path)
        
        # Visualize the sparse data to keep the plot looking clean!
        viz.draw_waypoints(waypoints_sparse)
        viz.draw_bubbles(bubbles_sparse)
        
        # Draw the final smooth path optimized entirely on the dense data
        viz.draw_optimized_path(P_opt)
        
    viz.show()

if __name__ == "__main__":
    main()