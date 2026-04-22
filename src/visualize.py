import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from .environment import GridEnvironment
import numpy as np


class EnvironmentVisualizer:
    def __init__(self, env: GridEnvironment):
        """Set up figure, axes, and checkbox panel. Don't draw yet."""
        self.env = env

        self.fig = plt.figure(figsize=(10,8))
        self.ax = self.fig.add_axes([0.1, 0.1, 0.65, 0.8])
        self.ax_check = self.fig.add_axes([0.8, 0.4, 0.15, 0.3])

        self.layers = {
            'Obstacles': [],
            'Start/Goal': [],
            'Gridlines': [],
            'A* Path': [],
            'Waypoints': [],
            'Bubbles': []
        }
    
    def _draw_obstacles(self) -> None:
        """Draw the occupancy grid. Store artists in self.layers['Obstacles']."""
        im = self.ax.imshow(
            self.env.occupancy_grid,
            cmap = 'Greys',
            origin = 'lower',
            extent = [0, self.env.width, 0, self.env.height],
            vmin = 0,
            vmax = 1,
        )
        self.layers['Obstacles'].append(im)
    
    def _draw_start_goal(self) -> None:
        """Draw start (green circle) and goal (red star) markers."""
        if self.env.start is not None:
            (line,) = self.ax.plot(
                self.env.start[0], self.env.start[1],
                marker='o', color = 'green', markersize = 15, label = 'Start',
            )
            self.layers['Start/Goal'].append(line)

        if self.env.goal is not None:
            (line,) = self.ax.plot(
                self.env.goal[0], self.env.goal[1],
                marker='*', color = 'red', markersize = 15, label = 'Goal',
            )
            self.layers['Start/Goal'].append(line)
    
    def _draw_gridlines(self) -> None:
        """Draw minor gridlines at every cell boundary."""
        for x in range(self.env.width +1):
            line = self.ax.axvline(x=x, color='gray', linewidth = 0.3, alpha = 0.5)
            self.layers['Gridlines'].append(line)

        for y in range(self.env.height +1):
            line = self.ax.axhline(y=y, color='gray', linewidth = 0.3, alpha = 0.5)
            self.layers['Gridlines'].append(line)
        
    
    def _on_toggle(self, label: str) -> None:
        """Called when a checkbox is clicked. Flip visibility of that layer."""
        for artist in self.layers[label]:
            current_state = artist.get_visible()
            artist.set_visible(not current_state)
        self.fig.canvas.draw_idle()
    
    def show(self) -> None:
        """Draw all layers, wire up checkboxes, display the figure."""
        
        #drawing methods
        self._draw_obstacles()
        self._draw_start_goal()
        self._draw_gridlines()

        #format plot area
        self.ax.set_xlim(0, self.env.width)
        self.ax.set_ylim(0, self.env.height)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')

        #buttons
        labels = list(self.layers.keys())
        actives = [True]*len(labels)
        self.check_buttons = CheckButtons(self.ax_check, labels, actives)
        self.check_buttons.on_clicked(self._on_toggle)

        plt.show()

    def draw_astar_path(self,path: list[tuple[int, int]]) -> None:
        if not path:
            return
        
        xs = [p[0] + 0.5 for p in path]
        ys = [p[1] + 0.5 for p in path]

        (line,) = self.ax.plot(
            xs, ys, color='blue', linewidth = 1.5, alpha = 0.7, label = 'A* Path'
        )
        self.layers['A* Path'].append(line)

    def draw_waypoints(self,waypoints: list[tuple[int,int]]) -> None:
        if not waypoints:
            return
        xs = [p[0] + 0.5 for p in waypoints]
        ys = [p[1] + 0.5 for p in waypoints]

        (line,) = self.ax.plot(
            xs, ys, color='orange', marker = 'o', linestyle = '--', linewidth = 1, alpha = 0.8, label = 'Waypoints'
        )
        self.layers['Waypoints'].append(line)

    def draw_bubbles(self, bubbles: list[tuple[float, float, float]]) -> None:
        """Draw safe circular bubbles around each waypoint"""
        from matplotlib.patches import Circle
        
        for cx, cy, r in bubbles:
            circle = Circle(
                (cx, cy),
                radius = r,
                fill = True,
                facecolor = 'cyan',
                edgecolor = 'blue',
                alpha = 0.15,
                linewidth = 0.8
            )
            self.ax.add_patch(circle)
            self.layers['Bubbles'].append(circle)

    def draw_optimized_path(self, P_opt, color='magenta', linewidth=2.5, label='Optimized path'):
        """Draw the SOCP-optimized smoothed path as a line."""
        P_opt = np.asarray(P_opt)
        self.ax.plot(
            P_opt[:, 0],
            P_opt[:, 1],
            '-',
            color=color,
            linewidth=linewidth,
            label=label,
            zorder=5,
        )