"""
test_corridor.py — Corridor-based SOCP motion planning pipeline.

This is an alternative to the bubble pipeline (test_environment.py). Instead
of safety circles (bubbles) around individual waypoints, it builds a sequence
of overlapping axis-aligned rectangular regions (a "corridor") covering the
entire path, then solves for the shortest smooth path inside that corridor.

Advantages over the bubble approach:
  - The corridor explicitly accounts for robot body size (obstacles inflated by
    robot_radius, so the center is planned as a point and the body is safe).
  - The path length objective is minimized directly (sum of segment norms),
    so the optimizer actively cuts corners wherever the corridor allows.
  - Corridor regions are larger than point bubbles, giving more room to optimize.

Full pipeline:
  1. Build the grid environment.
  2. Inflate each obstacle outward by robot_radius (Minkowski sum with a disk).
  3. Build a separate A* environment on the inflated grid, run A* with
     no-corner-cutting to get a discrete path safe for a disk-shaped robot.
  4. Downsample to corner waypoints; pin exact start/goal positions.
  5. Build the corridor: one axis-aligned polytope per A* segment, expanded
     outward into free space as far as possible.
  6. Choose how many optimizer points to use (based on path length / spacing).
  7. Assign each optimizer point to a corridor region (proportional to segment length).
  8. Solve the SOCP: minimize path length + smoothness s.t. corridor membership.
  9. Visualize: obstacles, inflated obstacles, A* path, corridor, optimized path,
     and robot body disks — all togglable.

Key tuning knobs:
  robot_radius     — size of the robot's disk body (affects obstacle inflation)
  desired_spacing  — target spacing between consecutive optimizer points
  smoothness_weight — 0 = pure length minimization, higher = rounder corners
  d_max            — cap on inter-point step distance (None = unconstrained)
"""

import numpy as np

from src.environment import GridEnvironment, RectObstacle
from src.astar import astar_search, downsample_collinear
from src.corridor import (
    inflate_obstacles,
    build_corridor,
    choose_n_points,
    assign_points_to_regions,
)
from src.corridor_optimizer import optimize_path_corridor
from src.visualize import EnvironmentVisualizer


def path_length(pts) -> float:
    """Total Euclidean length of a polyline."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def main():
    # ---- TUNING KNOBS ----
    robot_radius = 0.3        # size of the robot's disk body
    desired_spacing = 0.5     # spacing between optimization points (smaller = more points)
    smoothness_weight = 0.5   # 0 = pure length min; larger = rounder corners
    d_max = 2.0               # cap on inter-point step size (None to disable)

    # ---- 1. Build environment ----
    env = GridEnvironment(width=20, height=20)
    env.add_obstacle(RectObstacle(x_start=2,  y_start=0,  width=2, height=5))
    env.add_obstacle(RectObstacle(x_start=5,  y_start=3,  width=2, height=5))
    env.add_obstacle(RectObstacle(x_start=0,  y_start=9,  width=8, height=2))
    env.add_obstacle(RectObstacle(x_start=10, y_start=9,  width=10, height=2))
    env.add_obstacle(RectObstacle(x_start=3,  y_start=12, width=2, height=2))
    env.add_obstacle(RectObstacle(x_start=7,  y_start=14, width=2, height=2))
    env.add_obstacle(RectObstacle(x_start=11, y_start=16, width=2, height=2))
    env.add_obstacle(RectObstacle(x_start=15, y_start=13, width=4, height=2))
    env.add_obstacle(RectObstacle(x_start=15, y_start=15, width=2, height=4))
    env.add_obstacle(RectObstacle(x_start=14, y_start=3,  width=3, height=4))
    env.set_start(0.5, 1.5)
    env.set_goal(19.5, 17.5)

    # ---- 2. Inflate obstacles by the robot radius ----
    inflated = inflate_obstacles(env.obstacles, robot_radius)

    # ---- 3. Build a separate env for A* using the inflated obstacles ----
    # We clip to grid bounds because GridEnvironment.add_obstacle rejects
    # rectangles that extend outside the grid. The clipped rect still
    # blocks every cell whose center falls inside it -- which is all we
    # need from A*'s perspective.
    env_for_astar = GridEnvironment(width=env.width, height=env.height)
    for obs in inflated:
        x0 = max(0.0, obs.x_start)
        y0 = max(0.0, obs.y_start)
        x1 = min(env.width,  obs.x_start + obs.width)
        y1 = min(env.height, obs.y_start + obs.height)
        if x1 > x0 and y1 > y0:
            env_for_astar.add_obstacle(
                RectObstacle(x_start=x0, y_start=y0,
                             width=x1 - x0, height=y1 - y0)
            )
    env_for_astar.set_start(env.start[0], env.start[1])
    env_for_astar.set_goal(env.goal[0], env.goal[1])

    # ---- 4. Run A* with no-corner-cutting ----
    path = astar_search(env_for_astar, disallow_corner_cutting=True)
    if path is None:
        print("A* found no path. The environment may be over-inflated for r_robot.")
        return

    print(f"A* dense path: {len(path)} cells")

    corners = downsample_collinear(path)
    print(f"A* corners after downsampling: {len(corners)}")

    # World coordinates (cell centers), then pin exact start/goal
    corners_world = [(x + 0.5, y + 0.5) for x, y in corners]
    corners_world[0] = env.start
    corners_world[-1] = env.goal

    # ---- 5. Build the corridor ----
    corridor, validated = build_corridor(
        corners_world,
        inflated,
        env_bounds=(0.0, env.width, 0.0, env.height),
    )
    print(f"Corridor: {len(corridor)} polytopes "
          f"(validated waypoints: {len(validated)})")

    for i, poly in enumerate(corridor):
        print(f"  R{i}: x in [{poly.x_min:.2f}, {poly.x_max:.2f}], "
              f"y in [{poly.y_min:.2f}, {poly.y_max:.2f}], area={poly.area():.2f}")

    # ---- 6. Choose N optimization points and assign to regions ----
    n_points = choose_n_points(validated, desired_spacing=desired_spacing)
    assignment = assign_points_to_regions(n_points, corridor, validated)
    print(f"Optimization points: {n_points}")

    # ---- 7. Solve the SOCP ----
    P_opt = optimize_path_corridor(
        start=env.start,
        goal=env.goal,
        corridor=corridor,
        region_assignment=assignment,
        smoothness_weight=smoothness_weight,
        d_max=d_max,
    )

    # ---- Length comparison ----
    astar_world = [(x + 0.5, y + 0.5) for x, y in path]
    astar_world[0] = env.start
    astar_world[-1] = env.goal

    L_astar = path_length(astar_world)
    L_opt   = path_length(P_opt)
    print("\n--- Path length comparison ---")
    print(f"A* path length:        {L_astar:.3f}")
    print(f"Corridor-opt length:   {L_opt:.3f}")
    print(f"Reduction:             {(1 - L_opt/L_astar)*100:.1f}%")

    # ---- 8. Visualize ----
    viz = EnvironmentVisualizer(env)
    viz.draw_inflated_obstacles(inflated)
    viz.draw_astar_path(path)
    viz.draw_waypoints(corners)              # show the corner waypoints
    viz.draw_corridor(corridor)
    viz.draw_optimized_path(P_opt)
    viz.draw_robot_disks(P_opt, robot_radius)
    viz.show()


if __name__ == "__main__":
    main()