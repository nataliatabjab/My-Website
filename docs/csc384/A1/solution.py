#   Look for #IMPLEMENT tags in this file. These tags indicate what has
#   to be implemented to complete the warehouse domain.

#   You may add only standard python imports
#   You may not remove any imports.
#   You may not import or otherwise source any of your own files

import os  # for time functions
import math  # for infinity
from search import *  # for search engines
from sokoban import sokoban_goal_state, SokobanState, Direction, PROBLEMS  # for Sokoban specific classes and problems

# HELPER FUNCTIONS
# Helper function to find the minimum Manhattan distance from any robot to a valid push-square
def _robot_to_push_sq(state, box):
    """
    Returns the minimum Manhattan distance from any robot to a valid push-square
    adjacent to `box`.  A push-square is valid if:
      - the robot can stand there (not a wall or another box), and
      - the box can move into the opposite cell (not a wall or another box).
    If no valid push-square exists, returns math.inf.
    """
    (bx, by) = box
    best = math.inf

    # Check all four possible push directions
    for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        push_sq = (bx - dx, by - dy)
        dest_sq = (bx + dx, by + dy)

        # Check if the push square is a wall or another box
        if push_sq in state.obstacles or push_sq in state.boxes:
            continue
        if dest_sq in state.obstacles or dest_sq in state.boxes:
            continue

        # Check if the push square is reachable by any robot
        for (rx, ry) in state.robots:
            d = abs(rx - push_sq[0]) + abs(ry - push_sq[1])
            if d < best:
                best = d
    return best

# Helper function to find the minimum matching cost between boxes and storage
def _min_matching_cost(boxes, stores):
    """
    Return the exact minimum sum of Manhattan distances that pairs every
    box with a distinct storage square.  Assumes len(boxes) == len(stores).
    Uses O(2^n · n) DP; n ≤ 6 in A1 so it is instant.
    """
    m = len(boxes)
    dist = [[abs(bx - sx) + abs(by - sy) for (sx, sy) in stores]
            for (bx, by) in boxes]
    ALL = (1 << m) - 1
    dp = [math.inf] * (1 << m)
    dp[0] = 0
    for mask in range(ALL):
        i = bin(mask).count("1")
        for j in range(m):
            if mask & (1 << j):
                continue
            nxt = mask | (1 << j)
            dp[nxt] = min(dp[nxt], dp[mask] + dist[i][j])
    return dp[ALL]

# SOKOBAN HEURISTICS
def heur_alternate(state):
    '''a better heuristic'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.

    This heuristic improves upon the heur_manhattan_distance heuristic by detecting 
    deadlocks and computing a more accurate cost estimate to the goal. It does so by:

    - Returning `inf` immediately if the state is unsolvable due to:
        • A box stuck in a corner (unless it is on a storage square)
        • A box pushed against a wall with no reachable storage in that direction
        • A frozen 2x2 block of boxes (none on storage)
    - Otherwise, if no deadlock is detected, it computes:
        • The exact minimum matching cost between unmatched boxes and unmatched storage 
          (using a dynamic programming solver)
        • The minimum Manhattan distance from any robot to a valid push square for one of the boxes
    '''

    # Get the boxes and storage points
    boxes   = [b for b in state.boxes   if b not in state.storage]
    stores  = [s for s in state.storage if s not in state.boxes]
    if not boxes:
        return 0

    # Get the width and height of the grid
    W, H = state.width, state.height

    # Get the obstacles
    obstacles = state.obstacles

    # Get the box set
    box_set = set(boxes)

    for (x, y) in boxes:
        up    = (y-1 < 0)   or ((x, y-1) in obstacles)
        down  = (y+1 >= H)  or ((x, y+1) in obstacles)
        left  = (x-1 < 0)   or ((x-1, y) in obstacles)
        right = (x+1 >= W)  or ((x+1, y) in obstacles)

        # 1) simple corner
        if (up and left) or (up and right) or (down and left) or (down and right):
            return math.inf

        # 2) wall-hug with no storage beyond
        if (left and not any((sx, y) in state.storage for sx in range(x, W))) \
        or (right and not any((sx, y) in state.storage for sx in range(0, x+1))) \
        or (up and not any((x, sy) in state.storage for sy in range(y, H))) \
        or (down and not any((x, sy) in state.storage for sy in range(0, y+1))):
            return math.inf

        # 3) 2×2 frozen block (all four boxes, none on storage)
        if ((x+1, y) in box_set and (x, y+1) in box_set and (x+1, y+1) in box_set and
            all((cx, cy) not in state.storage for (cx, cy) in [(x, y), (x+1, y), (x, y+1), (x+1, y+1)])):
            return math.inf

    m = len(boxes)
    if len(stores) < m:
        return math.inf
    match_cost = _min_matching_cost(boxes, stores[:m])

    robot_push = min(_robot_to_push_sq(state, b) for b in boxes)
    if robot_push == math.inf:
        return math.inf

    return match_cost + robot_push


def heur_zero(state):
    '''Zero Heuristic can be used to make A* search perform uniform cost search'''
    return 0


def heur_manhattan_distance(state):
    '''admissible sokoban puzzle heuristic: manhattan distance'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''
    # We want an admissible heuristic, which is an optimistic heuristic.
    # It must never overestimate the cost to get from the current state to the goal.
    # The sum of the Manhattan distances between each box that has yet to be stored and the storage point nearest to it is such a heuristic.
    # When calculating distances, assume there are no obstacles on the grid.
    # You should implement this heuristic function exactly, even if it is tempting to improve it.
    # Your function should return a numeric value; this is the estimate of the distance to the goal.
    total_distance = 0
    boxes = state.boxes
    storage = state.storage

    # For each box, find the minimum distance to any storage point
    for box in boxes:

        # If box is in storage, skip it (distance is 0)
        if box in storage:
            continue

        # Find the minimum distance to any storage point
        min_dist = float('inf') # Start at infinity
        for storage_point in storage:

            # Calculate Manhattan distance: |x1 - x2| + |y1 - y2|
            dist = abs(box[0] - storage_point[0]) + abs(box[1] - storage_point[1])

            # If the distance is less than the current minimum, update the minimum
            min_dist = min(min_dist, dist)

        # Add the minimum distance to the total distance
        total_distance += min_dist
    return total_distance

def fval_function(sN, weight):
    """
    Provide a custom formula for f-value computation for Anytime Weighted A star.
    Returns the fval of the state contained in the sNode.

    @param sNode sN: A search node (containing a SokobanState)
    @param float weight: Weight given by Anytime Weighted A star
    @rtype: float
    """
    return sN.gval + weight * sN.hval

# SEARCH ALGORITHMS
def weighted_astar(initial_state, heur_fn, weight, timebound): 
    '''Provides an implementation of weighted a-star, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False as well as a SearchStats object'''
    '''implementation of weighted astar algorithm'''
    # 1) Create SearchEngine in "custom" mode
    se = SearchEngine('custom')
    # 2) Build f-value closure capturing weight
    fval_w = (lambda sN: fval_function(sN, weight))
    # 3) Initialize search with custom fval
    se.init_search(initial_state, sokoban_goal_state, heur_fn, fval_w)
    # 4) Run up to timebound
    final_state, stats = se.search(timebound)
    return final_state, stats


def iterative_astar(initial_state, heur_fn, weight=1, timebound=5):
    '''Provides an implementation of realtime a-star, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False as well as a SearchStats object'''
    '''implementation of iterative astar algorithm'''
    start_time = os.times()[0]
    best_state = False
    best_cost = float('inf')
    final_stats = None

    current_weight = weight
    consecutive_fails = 0  # Track consecutive failures
    
    while current_weight >= 1:
        remaining = timebound - (os.times()[0] - start_time)
        if remaining <= 0:
            break

        se = SearchEngine('custom')
        fval_w = (lambda sN, w_=current_weight: sN.gval + w_ * sN.hval)
        se.init_search(initial_state, sokoban_goal_state, heur_fn, fval_w)
        
        costbound = (float('inf'), float('inf'), best_cost)
        sol_state, stats = se.search(remaining, costbound)

        if sol_state:
            consecutive_fails = 0  # Reset failure counter
            if sol_state.gval < best_cost:
                best_cost = sol_state.gval
                best_state = sol_state
                final_stats = stats
                # More aggressive reduction when we find better solutions
                current_weight = max(1, current_weight * 0.75)
            else:
                # Less aggressive reduction when solution isn't better
                current_weight = max(1, current_weight * 0.85)
        else:
            consecutive_fails += 1
            if consecutive_fails > 2:
                # If failing repeatedly, reduce weight more cautiously
                current_weight = max(1, current_weight * 0.95)
            else:
                current_weight = max(1, current_weight * 0.9)
            
        if current_weight == 1:
            # One final attempt with remaining time at weight=1
            remaining = timebound - (os.times()[0] - start_time)
            if remaining > 0.1:  # Only if meaningful time remains
                se = SearchEngine('custom')
                fval_w = (lambda sN: sN.gval + sN.hval)
                se.init_search(initial_state, sokoban_goal_state, heur_fn, fval_w)
                final_try, stats = se.search(remaining, costbound)
                if final_try and final_try.gval < best_cost:
                    best_state = final_try
                    best_cost = final_try.gval
                    final_stats = stats
            break

    return best_state, final_stats


def iterative_gbfs(initial_state, heur_fn, timebound=5):
    '''Provides an implementation of anytime greedy best-first search, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False'''
    '''implementation of iterative gbfs algorithm'''

    start_time = os.times()[0]
    best_state = False
    best_cost = float('inf')
    final_stats = None

    while True:
        now = os.times()[0]
        elapsed = now - start_time
        remaining = timebound - elapsed
        if remaining <= 0:
            break

        # Use smaller time steps to allow more iterations
        slice_time = min(0.2 * remaining, remaining)

        se = SearchEngine('best_first')
        se.init_search(initial_state, sokoban_goal_state, heur_fn)

        # Use tighter cost bounds to focus search
        costbound = (best_cost * 0.95, float('inf'), float('inf'))
        sol_state, stats = se.search(slice_time, costbound)

        if not sol_state:
            # If no solution with current bounds, try again with relaxed bounds
            costbound = (best_cost, float('inf'), float('inf'))
            sol_state, stats = se.search(slice_time, costbound)
            if not sol_state:
                break

        if sol_state.gval < best_cost:
            best_cost = sol_state.gval
            best_state = sol_state
            final_stats = stats
        else:
            # If we found a solution but it's not better, we might be stuck
            break

    return best_state, final_stats