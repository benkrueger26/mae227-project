# Motion Planning Pipeline

A 2D motion planning pipeline that takes a robot from a start cell to a goal cell in a grid world with rectangular obstacles. Combines discrete A\* search with convex trajectory optimization to produce smooth, provably collision-free paths.

## Pipeline overview

The pipeline runs in four stages:

1. **A\* search** — finds a discrete shortest path on an 8-connected grid.
2. **Waypoint conditioning** — downsamples collinear runs from the A\* output, then resamples to a chosen spacing.
3. **Bubbles** — computes a circular safety region around each waypoint using exact point-to-rectangle distance, forming a continuous safety tube along the path.
4. **Convex optimization** — a Second-Order Cone Program (SOCP) re-positions the waypoints to minimize acceleration while staying inside the safety tube.

## Project structure

```
.
├── src/
│   ├── environment.py      # GridEnvironment + RectObstacle classes
│   ├── astar.py            # A* search and waypoint conditioning utilities
│   ├── bubbles.py          # Safety bubble computation
│   ├── optimizer.py        # SOCP trajectory smoother (CVXPY)
│   └── visualize.py        # Matplotlib visualization with toggleable layers
├── test_environment.py     # End-to-end demo / entry point
└── README.md
```

## Requirements

- Python 3.9+
- NumPy
- SciPy
- Matplotlib
- CVXPY
- Clarabel (SOCP solver, installed automatically with CVXPY)

## Setup

Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install numpy scipy matplotlib cvxpy clarabel
```

Optionally use a virtual environment first:

```bash
python -m venv venv
source venv/bin/activate    # on Windows: venv\Scripts\activate
pip install numpy scipy matplotlib cvxpy clarabel
```

## Running the demo

From the repo root:

```bash
python test_environment.py
```

This will:

1. Build a 20×20 grid world with several rectangular obstacles.
2. Print an ASCII view of the grid (start = `S`, goal = `G`, obstacles = `#`).
3. Run A\* and print path statistics at each conditioning stage.
4. Print diagnostic info comparing bubble radii against brute-force ground-truth distances.
5. Solve the SOCP optimization.
6. Print a path-length comparison between the raw A\* path and the optimized path.
7. Open a Matplotlib window showing the result.

## Visualization

The Matplotlib window has a checkbox panel on the right that lets you toggle individual layers on and off:

- **Obstacles** — the rasterized grid
- **Start/Goal** — green circle and red star
- **Gridlines** — minor cell boundaries
- **A\* Path** — raw A\* output (blue)
- **Waypoints** — sparse waypoints used for visualization (orange)
- **Bubbles** — safety circles (cyan)
- **Optimized Path** — final smoothed trajectory (magenta)

## Customizing the scene

Open `test_environment.py` and edit the `main()` function:

- **Grid size** — change `GridEnvironment(width=20, height=20)`.
- **Obstacles** — add or remove `env.add_obstacle(RectObstacle(...))` calls. Each rectangle is specified by its bottom-left corner, width, and height.
- **Start / goal** — `env.set_start(x, y)` and `env.set_goal(x, y)`. Use `(i + 0.5, j + 0.5)` to target the center of cell `(i, j)`.
- **Optimizer behavior** — in the call to `optimize_path(...)`:
  - `lambda_reg` controls how tightly the optimized path follows the A\* reference. `lambda_reg=0.0` lets the path pull taut like a rubber band; larger values keep it close to A\*.
  - `d_max` caps the distance between consecutive optimized points.
- **Waypoint density** — `resample_by_spacing(waypoints, max_spacing=0.25)` controls how dense the optimization input is. Tighter spacing means more, smaller bubbles but a smoother result.

## Notes

- Coordinates: the grid uses integer cell indices internally, but waypoints, bubbles, and the optimizer all work in continuous world coordinates. The `+0.5` offset in `test_environment.py` shifts cell indices to cell centers.
- The optimizer is convex, so the solver finds the global optimum on every run — no initialization sensitivity, no local minima.
- The bubbles guarantee safety only at the waypoints themselves; the `d_max` constraint is what keeps the line *segments* between consecutive points inside the safety tube as well.
