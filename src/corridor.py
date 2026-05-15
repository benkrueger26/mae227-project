"""
corridor.py — Safe convex corridor construction for 2D motion planning.

A "corridor" is a sequence of overlapping axis-aligned rectangular regions
(polytopes) that together form a collision-free tube from start to goal.
Every point inside a polytope is guaranteed to be in free space (accounting
for robot radius, since obstacles are pre-inflated before this step).

The optimizer then finds the smoothest path whose points each lie inside
their assigned polytope — since the polytopes are convex, this is an SOCP
(Second-Order Cone Program) with linear constraints and is solved globally.

Pipeline for this file:
  1. inflate_obstacles      — grow each obstacle by robot_radius so we can
                              plan the robot's center as a point-mass
  2. validate_and_split_waypoints — ensure no A* corner-to-corner segment
                              clips through an inflated obstacle edge
  3. build_polytope_for_segment — expand one axis-aligned box per segment
                              outward into free space as far as possible
  4. build_corridor         — call the above for every segment; return the
                              full corridor and the validated waypoint list
  5. assign_points_to_regions — decide which polytope each optimizer point
                              belongs to (proportional to segment length)
  6. choose_n_points        — pick total point count from path length and
                              desired spacing

Each polytope is stored as (x_min, x_max, y_min, y_max) bounds AND as
half-space form A @ x <= b, which is the format linear-constraint solvers
(including CVXPY) expect directly.
"""

from dataclasses import dataclass
import numpy as np
from .environment import RectObstacle


# ---------------------------------------------------------------------------
# Polytope dataclass
# ---------------------------------------------------------------------------

@dataclass
class Polytope:
    """Axis-aligned rectangular polytope stored in both bound and half-space form.

    Represents the set { (x, y) : x_min <= x <= x_max, y_min <= y <= y_max }.

    The A/b half-space form encodes the same set as four linear inequalities:
        A @ [x, y]^T <= b
    which is what CVXPY's linear constraint interface expects.

    Both forms are kept because:
      - bounds (x_min, x_max, ...) are easy to draw and reason about
      - (A, b) can be passed directly to the optimizer without conversion
    """
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def A(self) -> np.ndarray:
        """Half-space normal matrix, shape (4, 2).

        Each row is an outward-pointing normal of one face:
          row 0: [-1, 0] encodes  -x <= -x_min  i.e.  x >= x_min
          row 1: [ 1, 0] encodes   x <=  x_max
          row 2: [ 0,-1] encodes  -y <= -y_min  i.e.  y >= y_min
          row 3: [ 0, 1] encodes   y <=  y_max
        """
        return np.array([
            [-1.0,  0.0],
            [ 1.0,  0.0],
            [ 0.0, -1.0],
            [ 0.0,  1.0],
        ])

    @property
    def b(self) -> np.ndarray:
        """Half-space offset vector, shape (4,).

        Together with A, forms the constraint A @ [x, y]^T <= b.
        """
        return np.array([-self.x_min, self.x_max, -self.y_min, self.y_max])

    def contains(self, point: tuple[float, float], tol: float = 1e-9) -> bool:
        """Return True if the point lies inside this polytope (with tolerance).

        tol allows points that are numerically on the boundary to still pass.
        """
        x, y = point
        return (self.x_min - tol <= x <= self.x_max + tol and
                self.y_min - tol <= y <= self.y_max + tol)

    def area(self) -> float:
        """Area of the polytope rectangle."""
        return max(0.0, self.x_max - self.x_min) * max(0.0, self.y_max - self.y_min)


# ---------------------------------------------------------------------------
# Obstacle inflation
# ---------------------------------------------------------------------------

def inflate_obstacles(
    obstacles: list[RectObstacle],
    robot_radius: float,
) -> list[RectObstacle]:
    """Grow each obstacle outward by robot_radius on all four sides.

    Why: if we treat the robot as a point (its center), the robot's disk body
    collides with an obstacle whenever the center is within robot_radius of the
    obstacle surface. By inflating every obstacle by robot_radius first, we can
    then plan the center as a point-mass and still guarantee the body is safe.

    Note: inflated obstacles may extend outside the grid boundary — that is fine.
    The corridor builder and A* each clip or handle out-of-bounds separately.
    """
    inflated = []
    for obs in obstacles:
        inflated.append(RectObstacle(
            x_start=obs.x_start - robot_radius,
            y_start=obs.y_start - robot_radius,
            width=obs.width   + 2 * robot_radius,  # grow left AND right
            height=obs.height + 2 * robot_radius,  # grow bottom AND top
        ))
    return inflated


# ---------------------------------------------------------------------------
# Segment collision check (Liang-Barsky / slab method)
# ---------------------------------------------------------------------------

def _segment_intersects_obstacles(
    p0: tuple[float, float],
    p1: tuple[float, float],
    obstacles: list[RectObstacle],
    eps: float = 1e-9,
) -> bool:
    """Return True if the line segment p0->p1 enters the interior of any obstacle.

    Uses the slab method (Liang-Barsky algorithm): for each obstacle rectangle,
    find the range of t in [0, 1] for which the parametric point
        P(t) = p0 + t*(p1 - p0)
    lies inside the rectangle. If the t-intervals for the x-slab and y-slab
    overlap strictly, the segment penetrates the obstacle's interior.

    Boundary-only contact (tangency) returns False — the segment must actually
    enter the interior to count as an intersection.
    """
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0

    for obs in obstacles:
        ox0, oy0 = obs.x_start, obs.y_start
        ox1 = obs.x_start + obs.width
        oy1 = obs.y_start + obs.height

        # Start with the full parametric range [0, 1] and intersect it with
        # the x-slab and y-slab in turn. If the range becomes empty, the
        # segment doesn't penetrate this obstacle.
        t_enter, t_exit = 0.0, 1.0

        # --- X slab: ox0 <= P(t).x <= ox1 ---
        if abs(dx) < eps:
            # Segment is vertical. If x is outside the slab, no intersection.
            if x0 < ox0 + eps or x0 > ox1 - eps:
                continue
        else:
            t1 = (ox0 - x0) / dx
            t2 = (ox1 - x0) / dx
            t_lo, t_hi = (t1, t2) if t1 < t2 else (t2, t1)
            t_enter = max(t_enter, t_lo)
            t_exit  = min(t_exit,  t_hi)
            if t_enter >= t_exit:
                continue   # slab clipped the range to empty

        # --- Y slab: oy0 <= P(t).y <= oy1 ---
        if abs(dy) < eps:
            if y0 < oy0 + eps or y0 > oy1 - eps:
                continue
        else:
            t1 = (oy0 - y0) / dy
            t2 = (oy1 - y0) / dy
            t_lo, t_hi = (t1, t2) if t1 < t2 else (t2, t1)
            t_enter = max(t_enter, t_lo)
            t_exit  = min(t_exit,  t_hi)

        # Strict interior overlap: the t-ranges overlap by more than a point
        if t_exit - t_enter > eps:
            return True

    return False


# ---------------------------------------------------------------------------
# Waypoint validation (segment subdivision)
# ---------------------------------------------------------------------------

def validate_and_split_waypoints(
    waypoints: list[tuple[float, float]],
    inflated_obstacles: list[RectObstacle],
    max_recursion: int = 6,
) -> list[tuple[float, float]]:
    """Ensure every consecutive segment of waypoints is collision-free.

    Problem: A* visits discrete grid cells, and the corner-downsampled path
    connects those corners with straight line segments. Even though every cell
    visited by A* is free, a diagonal corner-to-corner segment can geometrically
    clip the corner of an inflated obstacle. This would cause the polytope builder
    to fail (it can't contain both endpoints in a single obstacle-free box).

    Fix: for each bad segment, recursively insert the midpoint and retest.
    Each subdivision halves the segment length, so after a few rounds the segment
    is shorter than any obstacle face and guaranteed to clear it.

    max_recursion: safety limit on subdivision depth. At depth 6, segments are
    reduced to 1/64 of their original length — shorter than any reasonable obstacle.
    """
    if len(waypoints) < 2:
        return waypoints[:]

    def split(p0, p1, depth):
        # Base case: segment is already collision-free
        if not _segment_intersects_obstacles(p0, p1, inflated_obstacles):
            return [p0]
        # Depth limit: stop subdividing even if still colliding
        if depth >= max_recursion:
            return [p0]
        # Subdivide: insert the midpoint and recurse on both halves
        mid = (0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1]))
        return split(p0, mid, depth + 1) + split(mid, p1, depth + 1)

    out = []
    for i in range(len(waypoints) - 1):
        out.extend(split(waypoints[i], waypoints[i + 1], 0))
    out.append(waypoints[-1])   # always include the final endpoint
    return out


# ---------------------------------------------------------------------------
# Single-segment polytope builder (face expansion)
# ---------------------------------------------------------------------------

def _point_seed_box(
    point: tuple[float, float],
    seed_margin: float = 1e-3,
) -> tuple[float, float, float, float]:
    """Create a tiny axis-aligned box centered on `point`.

    This is the guaranteed-free seed from which face expansion starts.
    Since `point` is on a validated (collision-free) A* segment, a
    sufficiently small box around it is also in free space.

    Returns (x_min, x_max, y_min, y_max).
    """
    x, y = point
    return (x - seed_margin, x + seed_margin,
            y - seed_margin, y + seed_margin)


def _expand_face(
    face: str,
    box: tuple[float, float, float, float],
    obstacles: list[RectObstacle],
    env_bounds: tuple[float, float, float, float],
) -> float:
    """Push one face of `box` outward as far as possible without hitting obstacles.

    Strategy: start with the environment boundary as the target. Then for every
    obstacle whose projection onto the perpendicular axis overlaps the box's
    current extent, check whether it blocks the expansion. If so, retract the
    target to the obstacle's near face. The tightest such limit wins.

    Parameters
    ----------
    face      : 'left', 'right', 'bottom', or 'top'
    box       : current (x_min, x_max, y_min, y_max) of the polytope seed
    obstacles : inflated obstacle list
    env_bounds: (x_lo, x_hi, y_lo, y_hi) of the planning environment

    Returns the new value for the expanded face coordinate.
    """
    x_min, x_max, y_min, y_max = box
    x_lo, x_hi, y_lo, y_hi = env_bounds

    if face == 'left':
        # Move x_min toward x_lo (expand leftward).
        # An obstacle blocks if its y-extent overlaps [y_min, y_max] AND its
        # right edge (o_x1) is to the left of the current x_min.
        limit = x_lo
        for obs in obstacles:
            o_x1 = obs.x_start + obs.width
            o_y0, o_y1 = obs.y_start, obs.y_start + obs.height
            if o_y1 <= y_min or o_y0 >= y_max:
                continue    # no y-overlap, doesn't block
            if o_x1 <= x_min:
                limit = max(limit, o_x1)    # retract to this obstacle's right edge
        return limit

    if face == 'right':
        # Move x_max toward x_hi (expand rightward).
        limit = x_hi
        for obs in obstacles:
            o_x0 = obs.x_start
            o_y0, o_y1 = obs.y_start, obs.y_start + obs.height
            if o_y1 <= y_min or o_y0 >= y_max:
                continue
            if o_x0 >= x_max:
                limit = min(limit, o_x0)    # retract to this obstacle's left edge
        return limit

    if face == 'bottom':
        # Move y_min toward y_lo (expand downward).
        limit = y_lo
        for obs in obstacles:
            o_x0, o_x1 = obs.x_start, obs.x_start + obs.width
            o_y1 = obs.y_start + obs.height
            if o_x1 <= x_min or o_x0 >= x_max:
                continue
            if o_y1 <= y_min:
                limit = max(limit, o_y1)
        return limit

    if face == 'top':
        # Move y_max toward y_hi (expand upward).
        limit = y_hi
        for obs in obstacles:
            o_x0, o_x1 = obs.x_start, obs.x_start + obs.width
            o_y0 = obs.y_start
            if o_x1 <= x_min or o_x0 >= x_max:
                continue
            if o_y0 >= y_max:
                limit = min(limit, o_y0)
        return limit

    raise ValueError(f"Unknown face: {face}")


def build_polytope_for_segment(
    p0: tuple[float, float],
    p1: tuple[float, float],
    inflated_obstacles: list[RectObstacle],
    env_bounds: tuple[float, float, float, float],
    seed_margin: float = 1e-3,
) -> Polytope:
    """Build a single obstacle-free axis-aligned polytope around segment p0->p1.

    Algorithm:
      1. Seed a tiny box at the segment's midpoint (guaranteed in free space).
      2. Expand each face outward one at a time, stopping at the nearest
         obstacle or environment boundary.
      3. Assert that both p0 and p1 are inside the resulting polytope.
         (If they aren't, validate_and_split_waypoints should have been called
         first to ensure the segment doesn't clip any obstacle.)

    Expansion order (left, right, bottom, top) is sequential: each expansion
    uses the latest box state, so a wide horizontal expansion gives more room
    for the vertical expansion to follow.
    """
    midpoint = (0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1]))
    x_min, x_max, y_min, y_max = _point_seed_box(midpoint, seed_margin)

    # Expand each face greedily into free space
    x_min = _expand_face('left',   (x_min, x_max, y_min, y_max), inflated_obstacles, env_bounds)
    x_max = _expand_face('right',  (x_min, x_max, y_min, y_max), inflated_obstacles, env_bounds)
    y_min = _expand_face('bottom', (x_min, x_max, y_min, y_max), inflated_obstacles, env_bounds)
    y_max = _expand_face('top',    (x_min, x_max, y_min, y_max), inflated_obstacles, env_bounds)

    poly = Polytope(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    # Safety check: both segment endpoints must be inside the polytope.
    # Failure here means the segment clipped an obstacle — run
    # validate_and_split_waypoints on the corner list before calling this.
    if not (poly.contains(p0) and poly.contains(p1)):
        raise ValueError(
            f"Polytope for segment {p0}->{p1} does not contain both endpoints. "
            f"Got box [{x_min:.3f},{x_max:.3f}]x[{y_min:.3f},{y_max:.3f}]. "
            f"Run validate_and_split_waypoints on the corner list first."
        )

    return poly


# ---------------------------------------------------------------------------
# Full corridor builder
# ---------------------------------------------------------------------------

def build_corridor(
    waypoints: list[tuple[float, float]],
    inflated_obstacles: list[RectObstacle],
    env_bounds: tuple[float, float, float, float],
) -> tuple[list[Polytope], list[tuple[float, float]]]:
    """Build a sequence of obstacle-free polytopes covering all path segments.

    For N waypoints, this produces N-1 polytopes — one per consecutive pair.
    Adjacent polytopes overlap (they both contain the shared waypoint between
    their segments), so the optimizer can smoothly transition between them.

    Parameters
    ----------
    waypoints         : corner-downsampled A* path in world coords (x+0.5, y+0.5)
    inflated_obstacles: obstacles already grown by robot_radius
    env_bounds        : (x_lo, x_hi, y_lo, y_hi) of the planning environment

    Returns
    -------
    (corridor, validated_waypoints)
      corridor           : list of Polytope, one per segment
      validated_waypoints: waypoint list after subdivision — may be longer than
                           the input; this is the list the optimizer uses for
                           region assignment.
    """
    if len(waypoints) < 2:
        raise ValueError("Need at least 2 waypoints to build a corridor.")

    # Subdivide any segments that geometrically clip an inflated obstacle
    validated = validate_and_split_waypoints(waypoints, inflated_obstacles)

    # Build one polytope per consecutive pair of validated waypoints
    corridor = []
    for i in range(len(validated) - 1):
        poly = build_polytope_for_segment(
            validated[i], validated[i + 1],
            inflated_obstacles, env_bounds,
        )
        corridor.append(poly)

    return corridor, validated


# ---------------------------------------------------------------------------
# Point count and region assignment
# ---------------------------------------------------------------------------

def choose_n_points(
    waypoints: list[tuple[float, float]],
    desired_spacing: float,
    min_points: int = 8,
) -> int:
    """Choose how many optimizer points to use based on path length.

    Total path length divided by desired_spacing gives the number of points
    needed to achieve roughly that spacing between consecutive optimizer points.
    min_points ensures there are always enough points for the smoothness term
    to have something to work with even on very short paths.
    """
    total_length = 0.0
    for i in range(len(waypoints) - 1):
        dx = waypoints[i + 1][0] - waypoints[i][0]
        dy = waypoints[i + 1][1] - waypoints[i][1]
        total_length += float(np.hypot(dx, dy))
    n = int(np.ceil(total_length / desired_spacing)) + 1
    return max(min_points, n)


def assign_points_to_regions(
    n_points: int,
    corridor: list[Polytope],
    waypoints: list[tuple[float, float]],
) -> list[int]:
    """Decide which polytope (corridor region) each optimizer point belongs to.

    Each optimizer point must be constrained to lie in exactly one polytope.
    Points are distributed across regions proportional to segment length:
    longer segments get more optimizer points, giving roughly uniform spacing.

    The start point (index 0) is always in region 0, and the end point
    (index n_points-1) is always in the last region.

    Returns
    -------
    List of length n_points. Entry i is the polytope index for optimizer point i.
    """
    n_regions = len(corridor)
    assert n_regions == len(waypoints) - 1, "Need one region per segment."

    # Compute each segment's contribution to the total path length
    seg_lengths = []
    for i in range(n_regions):
        p0 = np.array(waypoints[i])
        p1 = np.array(waypoints[i + 1])
        seg_lengths.append(float(np.linalg.norm(p1 - p0)))
    total = sum(seg_lengths)

    if total == 0:
        # Degenerate: all waypoints at the same location — distribute evenly
        seg_lengths = [1.0] * n_regions
        total = float(n_regions)

    # Allocate point counts proportional to segment length; minimum 1 per region
    raw = [max(1, int(round(n_points * L / total))) for L in seg_lengths]

    # Adjust the allocation to sum to exactly n_points by adding/removing
    # from the region with the most "room" (relative to its segment length)
    diff = n_points - sum(raw)
    while diff != 0:
        if diff > 0:
            idx = max(range(n_regions), key=lambda k: seg_lengths[k] / raw[k])
            raw[idx] += 1
            diff -= 1
        else:
            candidates = [k for k in range(n_regions) if raw[k] > 1]
            if not candidates:
                break
            idx = max(candidates, key=lambda k: raw[k])
            raw[idx] -= 1
            diff += 1

    # Flatten: [region 0] * count[0] + [region 1] * count[1] + ...
    assignment = []
    for region_idx, count in enumerate(raw):
        assignment.extend([region_idx] * count)
    return assignment
