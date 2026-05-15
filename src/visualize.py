"""
visualize.py — Interactive matplotlib visualizer for the motion planning pipeline.

EnvironmentVisualizer draws all planning artifacts (grid, obstacles, A* path,
waypoints, bubbles, corridor, optimized path, robot disks) as separate named
layers. Each layer is wired to a CheckButton so the user can toggle it on/off
at runtime to understand what each stage of the pipeline contributes.

Layer toggling works by storing every matplotlib Artist in a dict keyed by
layer name. When a checkbox is clicked, _on_toggle flips the visibility of
every artist in that layer.

Usage:
    viz = EnvironmentVisualizer(env)
    viz.draw_astar_path(path)
    viz.draw_bubbles(bubbles)
    viz.draw_optimized_path(P_opt)
    viz.show()    # blocks until the window is closed
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from matplotlib.patches import Circle, Rectangle
from .environment import GridEnvironment
import numpy as np


class EnvironmentVisualizer:
    def __init__(self, env: GridEnvironment):
        """Set up the figure, main axes, and checkbox panel.

        Nothing is drawn yet — call the draw_* methods first, then show().
        The checkbox panel is sized to accommodate all layers.
        """
        self.env = env

        self.fig = plt.figure(figsize=(10, 8))
        # Main plotting area: left 65% of the figure
        self.ax = self.fig.add_axes([0.1, 0.1, 0.65, 0.8])
        # Checkbox panel: right strip, taller to fit all layer labels
        self.ax_check = self.fig.add_axes([0.8, 0.3, 0.15, 0.45])

        # Each key is a layer name shown in the checkbox panel.
        # Values are lists of matplotlib Artists. When a checkbox is toggled,
        # every artist in that layer has its visibility flipped.
        self.layers = {
            'Obstacles':          [],   # imshow of the occupancy grid
            'Inflated Obstacles': [],   # dashed red rectangles (robot-radius grown)
            'Start/Goal':         [],   # green circle (S) and red star (G)
            'Gridlines':          [],   # light gray cell boundary lines
            'A* Path':            [],   # raw A* path (blue line, 4-connected)
            'Waypoints':          [],   # corner waypoints (orange dots/dashes)
            'Bubbles':            [],   # cyan safety circles (bubble pipeline)
            'Corridor':           [],   # green rectangles (corridor pipeline)
            'Optimized Path':     [],   # smooth magenta path from SOCP
            'Robot Disks':        [],   # magenta circles showing robot body size
        }

    # -----------------------------------------------------------------------
    # Internal drawing helpers (called by show())
    # -----------------------------------------------------------------------

    def _draw_obstacles(self) -> None:
        """Render the occupancy grid as a grayscale image.

        imshow with origin='lower' matches the (x-right, y-up) convention
        used throughout the planner. extent=[0, W, 0, H] maps pixel edges
        to world coordinates so obstacles line up with the integer grid lines.
        """
        im = self.ax.imshow(
            self.env.occupancy_grid,
            cmap='Greys',
            origin='lower',
            extent=[0, self.env.width, 0, self.env.height],
            vmin=0,
            vmax=1,
        )
        self.layers['Obstacles'].append(im)

    def _draw_start_goal(self) -> None:
        """Draw start (green circle) and goal (red star) markers."""
        if self.env.start is not None:
            (line,) = self.ax.plot(
                self.env.start[0], self.env.start[1],
                marker='o', color='green', markersize=15, label='Start',
            )
            self.layers['Start/Goal'].append(line)

        if self.env.goal is not None:
            (line,) = self.ax.plot(
                self.env.goal[0], self.env.goal[1],
                marker='*', color='red', markersize=15, label='Goal',
            )
            self.layers['Start/Goal'].append(line)

    def _draw_gridlines(self) -> None:
        """Draw faint gray lines at every integer cell boundary."""
        for x in range(self.env.width + 1):
            line = self.ax.axvline(x=x, color='gray', linewidth=0.3, alpha=0.5)
            self.layers['Gridlines'].append(line)
        for y in range(self.env.height + 1):
            line = self.ax.axhline(y=y, color='gray', linewidth=0.3, alpha=0.5)
            self.layers['Gridlines'].append(line)

    def _on_toggle(self, label: str) -> None:
        """Checkbox callback: flip visibility of every artist in the named layer."""
        for artist in self.layers[label]:
            artist.set_visible(not artist.get_visible())
        self.fig.canvas.draw_idle()   # redraw without blocking

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def show(self) -> None:
        """Draw the base layers (obstacles, start/goal, gridlines), wire up
        the checkboxes, format the axes, and display the figure.

        Call this after all draw_* methods. Blocks until the window is closed.
        """
        self._draw_obstacles()
        self._draw_start_goal()
        self._draw_gridlines()

        self.ax.set_xlim(0, self.env.width)
        self.ax.set_ylim(0, self.env.height)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')

        # Build checkboxes from the layer dict. All layers start visible (True).
        labels = list(self.layers.keys())
        actives = [True] * len(labels)
        self.check_buttons = CheckButtons(self.ax_check, labels, actives)
        self.check_buttons.on_clicked(self._on_toggle)

        plt.show()

    def draw_astar_path(self, path: list[tuple[int, int]]) -> None:
        """Draw the raw A* grid path as a blue polyline.

        path contains integer cell indices (x, y). Adding 0.5 shifts each
        point to the cell center in world coordinates, which aligns with the
        continuous-space convention used everywhere else.
        """
        if not path:
            return
        xs = [p[0] + 0.5 for p in path]
        ys = [p[1] + 0.5 for p in path]
        (line,) = self.ax.plot(
            xs, ys, color='blue', linewidth=1.5, alpha=0.7, label='A* Path'
        )
        self.layers['A* Path'].append(line)

    def draw_waypoints(self, waypoints: list[tuple[int, int]]) -> None:
        """Draw the corner waypoints as an orange dashed line with dot markers.

        waypoints are integer cell indices from downsample_collinear, so +0.5
        converts them to world (cell-center) coordinates.
        """
        if not waypoints:
            return
        xs = [p[0] + 0.5 for p in waypoints]
        ys = [p[1] + 0.5 for p in waypoints]
        (line,) = self.ax.plot(
            xs, ys, color='orange', marker='o', linestyle='--',
            linewidth=1, alpha=0.8, label='Waypoints'
        )
        self.layers['Waypoints'].append(line)

    def draw_bubbles(self, bubbles: list[tuple[float, float, float]]) -> None:
        """Draw a cyan semi-transparent circle for each safety bubble.

        Each bubble is (cx, cy, r) in world coordinates. Circles are added as
        patches rather than plot artists — each is stored individually so the
        layer toggle can show/hide all of them at once.
        """
        for cx, cy, r in bubbles:
            circle = Circle(
                (cx, cy),
                radius=r,
                fill=True,
                facecolor='cyan',
                edgecolor='blue',
                alpha=0.15,
                linewidth=0.8,
            )
            self.ax.add_patch(circle)
            self.layers['Bubbles'].append(circle)

    def draw_optimized_path(
        self, P_opt, color='magenta', linewidth=2.5, label='Optimized path'
    ) -> None:
        """Draw the SOCP-optimized smooth path as a magenta line.

        P_opt is a (K, 2) numpy array of world coordinates. The line artist
        is stored in layers['Optimized Path'] so the checkbox can toggle it.
        """
        P_opt = np.asarray(P_opt)
        (line,) = self.ax.plot(
            P_opt[:, 0], P_opt[:, 1],
            '-', color=color, linewidth=linewidth, label=label, zorder=5,
        )
        self.layers['Optimized Path'].append(line)

    def draw_inflated_obstacles(self, inflated_obstacles) -> None:
        """Draw robot-radius-inflated obstacles as dashed red rectangles.

        These rectangles show the forbidden region for the robot's center point.
        Staying outside them guarantees the disk-shaped robot body won't collide
        with any original obstacle.
        """
        for obs in inflated_obstacles:
            rect = Rectangle(
                (obs.x_start, obs.y_start),
                obs.width,
                obs.height,
                facecolor='red',
                alpha=0.12,
                edgecolor='red',
                linewidth=0.7,
                linestyle='--',
                zorder=1,
            )
            self.ax.add_patch(rect)
            self.layers['Inflated Obstacles'].append(rect)

    def draw_corridor(self, corridor) -> None:
        """Draw corridor polytopes as semi-transparent green rectangles.

        Each polytope is a collision-free region. Where consecutive polytopes
        overlap, the region appears slightly darker green (alpha stacking).
        The optimizer constrains each path point to stay inside its polytope.
        """
        for poly in corridor:
            rect = Rectangle(
                (poly.x_min, poly.y_min),
                poly.x_max - poly.x_min,
                poly.y_max - poly.y_min,
                facecolor='lightgreen',
                alpha=0.18,
                edgecolor='green',
                linewidth=0.8,
                zorder=2,
            )
            self.ax.add_patch(rect)
            self.layers['Corridor'].append(rect)

    def draw_robot_disks(
        self,
        P_opt,
        robot_radius: float,
        stride: int = None,
        color: str = 'magenta',
    ) -> None:
        """Draw the robot's disk body at sampled positions along the optimized path.

        Visualizing the robot body at several positions lets you verify by eye that
        the disk never overlaps any obstacle — i.e. that the safety guarantee holds.

        stride : draw a disk every `stride` optimizer points. Auto-picks ~15 disks
                 if not specified.
        """
        P_opt = np.asarray(P_opt)
        n = len(P_opt)
        if stride is None:
            stride = max(1, n // 15)   # aim for roughly 15 disk snapshots

        for k in range(0, n, stride):
            disk = Circle(
                (P_opt[k, 0], P_opt[k, 1]),
                radius=robot_radius,
                facecolor=color,
                alpha=0.15,
                edgecolor=color,
                linewidth=0.8,
                zorder=4,
            )
            self.ax.add_patch(disk)
            self.layers['Robot Disks'].append(disk)
