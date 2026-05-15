# Motion Planning Pipeline

A 2D motion planning pipeline that takes a robot from a start cell to a goal cell in a grid world with rectangular obstacles. Combines discrete A\* search with convex trajectory optimization (SOCP) to produce smooth, provably collision-free paths.

Two complete pipelines are implemented:

| Pipeline | Entry point | Approach |
|---|---|---|
| **Bubble** | `test_environment.py` | Safety circles around each waypoint |
| **Corridor** | `test_corridor.py` | Safe rectangular corridor along the full path |

---

## How it works

### Bubble pipeline (`test_environment.py`)

1. **A\* search** — finds a discrete shortest path on an 8-connected grid.
2. **Waypoint conditioning** — downsamples collinear runs, then resamples to a dense uniform spacing.
3. **Safety bubbles** — computes a circular safe region around each waypoint using exact point-to-rectangle distance. Together the circles form a continuous collision-free tube.
4. **SOCP optimization** — re-positions the waypoints to minimize path acceleration (smoothness) while keeping every point inside its bubble. With `lambda_reg=0` the path pulls taut like a rubber band inside the tube.

### Corridor pipeline (`test_corridor.py`)

1. **Obstacle inflation** — every obstacle is grown outward by `robot_radius` (Minkowski sum). Planning the robot's center point against the inflated obstacles is equivalent to planning the full disk body against the originals.
2. **A\* on the inflated grid** — run with no-corner-cutting to ensure the discrete path is safe for a disk-shaped body.
3. **Corridor construction** — one axis-aligned rectangular polytope is built per path segment, expanded outward into free space as far as possible. Consecutive polytopes overlap, forming a continuous tube.
4. **SOCP optimization** — minimizes path length plus a smoothness penalty, subject to each optimizer point staying inside its assigned corridor rectangle. The path actively cuts corners wherever the corridor allows.

---

## Project structure

```
.
├── src/
│   ├── environment.py        # GridEnvironment + RectObstacle classes
│   ├── astar.py              # A* search, downsample_collinear, resample_by_spacing
│   ├── bubbles.py            # Distance field + safety bubble computation
│   ├── optimizer.py          # Bubble-constrained SOCP smoother (CVXPY)
│   ├── corridor.py           # Obstacle inflation, corridor construction, region assignment
│   ├── corridor_optimizer.py # Corridor-constrained SOCP optimizer (CVXPY)
│   └── visualize.py          # Matplotlib visualizer with toggleable layers
├── test_environment.py       # Bubble pipeline demo
├── test_corridor.py          # Corridor pipeline demo
└── README.md
```

---

## Requirements

- Python 3.10+
- NumPy
- SciPy
- Matplotlib
- CVXPY + Clarabel (SOCP solver, installed automatically with CVXPY)

---

## Setup

```bash
git clone https://github.com/benkrueger26/mae227-project.git
cd mae227-project
pip install numpy scipy matplotlib cvxpy
```

Using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install numpy scipy matplotlib cvxpy
```

---

## Running the demos

From the repo root:

```bash
# Bubble pipeline
python test_environment.py

# Corridor pipeline
python test_corridor.py
```

Both scripts will:
1. Build a 20×20 grid world with several rectangular obstacles.
2. Print an ASCII map (`S` = start, `G` = goal, `#` = obstacle).
3. Run A\* and print path statistics.
4. Solve the SOCP and print path length before and after optimization.
5. Open a Matplotlib window with all planning layers.

---

## Visualization

The Matplotlib window has a checkbox panel that toggles individual layers on/off:

| Layer | Description |
|---|---|
| Obstacles | Rasterized occupancy grid |
| Inflated Obstacles | Robot-radius-grown obstacles (corridor pipeline) |
| Start / Goal | Green circle (S) and red star (G) |
| Gridlines | Minor cell boundary lines |
| A\* Path | Raw discrete A\* output (blue) |
| Waypoints | Sparse corner waypoints (orange) |
| Bubbles | Safety circles (cyan) — bubble pipeline |
| Corridor | Safe rectangular regions (green) — corridor pipeline |
| Optimized Path | Final smoothed trajectory (magenta) |
| Robot Disks | Robot body at sampled positions — verifies no collision |

---

## Customizing the scene

Edit the `main()` function in either test file:

**Grid and obstacles**
```python
env = GridEnvironment(width=20, height=20)
env.add_obstacle(RectObstacle(x_start=2, y_start=0, width=2, height=5))
```
Each `RectObstacle` is specified by its bottom-left corner `(x_start, y_start)`, `width`, and `height` in world units.

**Start and goal**
```python
env.set_start(0.5, 1.5)   # center of cell (0, 1)
env.set_goal(19.5, 17.5)  # center of cell (19, 17)
```
Use `(i + 0.5, j + 0.5)` to target the center of cell `(i, j)`.

**Bubble pipeline tuning**
```python
waypoints_dense = resample_by_spacing(waypoints, max_spacing=0.25)  # point density
P_opt = optimize_path(q, bubbles, lambda_reg=0.0, d_max=1.5)
```
- `max_spacing` — smaller = more optimizer points = smoother result
- `lambda_reg` — `0.0` = pure rubber-band; larger = path stays closer to A\*
- `d_max` — caps step length between consecutive points; prevents bunching

**Corridor pipeline tuning**
```python
robot_radius     = 0.3   # disk body radius
desired_spacing  = 0.5   # target spacing between optimizer points
smoothness_weight = 0.5  # 0 = pure length minimization, higher = rounder corners
d_max            = 2.0   # step-length cap (None to disable)
```

---

## Key concepts

**Why SOCP?**
All constraints (bubble radii, step distances, corridor rectangles) are either linear or second-order cone constraints. CVXPY recognizes this and passes the problem to the CLARABEL solver, which finds the provably global optimum — no local minima, no sensitivity to initialization.

**Coordinate convention**
Integer cell `(i, j)` occupies the unit square from `(i, j)` to `(i+1, j+1)`. Its center is at `(i+0.5, j+0.5)`. Waypoints, bubbles, the corridor, and the optimizer all work in continuous world coordinates.

**Safety guarantee**
For the bubble pipeline: the bubble radius at each waypoint equals the true distance to the nearest obstacle minus a safety margin. Any point inside the bubble is collision-free. The `d_max` constraint ensures that the path segments between bubbles also stay within the tube.

For the corridor pipeline: every optimizer point is constrained to a rectangle that was constructed to be entirely in free space (after obstacle inflation). The robot body is safe everywhere along the path by construction.
