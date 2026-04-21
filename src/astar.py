# src/astar.py
import heapq
import math
from typing import Optional
from .environment import GridEnvironment


def euclidean_heuristic(state: tuple[int, int], goal: tuple[int, int]) -> float:
    """Straight-line distance from state to goal. Admissible for 8-connected grids."""
    return math.hypot(goal[0]-state[0], goal[1] - state[1])

def _reconstruct_path(came_from: dict, goal: tuple[int, int]) -> list[tuple[int,int]]:
    #Walk backward from goal to start using backpointers,  then reverse
    path = [goal]
    #Keep tracing back until we hit the start node
    while came_from[path[-1]] is not None:
        path.append(came_from[path[-1]])

    #Want the path to go from the start to the goal
    path.reverse()
    return path

def astar_search(
    env: GridEnvironment,
    heuristic=euclidean_heuristic,
) -> Optional[list[tuple[int, int]]]:
    """
    Run A* from env.start to env.goal on an 8-connected grid.
    Returns the full path as a list of (x, y) integer cells, or None if no path exists.
    """
    if env.start is None or env.goal is None:
        return None
    
    start = (int(env.start[0]), int(env.start[1]))
    goal = (int(env.goal[0]), int(env.goal[1]))

    #Define legal moves (up, down, right left, and then aslo the diagonal directiosn)
    moves = [
        (1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)
    ]

    #Priotrity Queue, f = g+h, entriies are f_score, state
    frontier = []
    heapq.heappush(frontier, (heuristic(start,goal), start))

    #Best score to reach each state
    g_score = {start:0.0}

    #Backpointers
    came_from = {start:None}

    while frontier:
        f_current, current = heapq.heappop(frontier)

        if current == goal:
            return _reconstruct_path(came_from, goal)
        
        for dx, dy in moves:
            neighbor = (current[0] + dx, current[1] + dy)

            if not env.is_free(neighbor[0], neighbor[1]):
                continue
            
            step_cost = math.hypot(dx,dy)
            tentative_g = g_score[current] + step_cost

            if tentative_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current

                f_neighbor = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(frontier, (f_neighbor, neighbor))

    return None

def downsample_collinear(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Keep only waypoints where the direction of motion changes.
    Input: dense path from A*. Output: corners-only path.
    """
    if not path or len(path) <= 2:
        return path[:]
    
    result = [path[0]]

    for i in range(1, len(path) - 1):
        prev = path[i-1]
        curr = path[i]
        nxt = path[i+1]

        # calculate directional vectors
        dir_in = (curr[0]- prev[0], curr[1]- prev[1])
        dir_out = (nxt[0] - curr[0], nxt[1] - curr[1])

        if dir_in != dir_out:
            result.append(curr)

    result.append(path[-1])
    return result