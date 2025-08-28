#   Look for #IMPLEMENT tags in this file. These tags indicate what has
#   to be implemented to complete the warehouse domain.

#   You may add only standard python imports
#   You may not remove any imports.
#   You may not import or otherwise source any of your own files

import os  # for time functions
import math  # for infinity
from search import *  # for search engines
from sokoban import sokoban_goal_state, SokobanState, Direction, PROBLEMS  # for Sokoban specific classes and problems

from itertools import permutations
import math

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
    for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        push_sq = (bx - dx, by - dy)
        dest_sq = (bx + dx, by + dy)
        if push_sq in state.obstacles or push_sq in state.boxes:
            continue
        if dest_sq in state.obstacles or dest_sq in state.boxes:
            continue
        for (rx, ry) in state.robots:
            d = abs(rx - push_sq[0]) + abs(ry - push_sq[1])
            if d < best:
                best = d
    return best

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
        i = bin(mask).count("1")           # how many boxes already matched
        for j in range(m):
            if mask & (1 << j):
                continue
            nxt = mask | (1 << j)
            dp[nxt] = min(dp[nxt], dp[mask] + dist[i][j])
    return dp[ALL]

# SOKOBAN HEURISTICS
def heur_alternate(state):
    """
    Heuristic =  dead-lock test  OR  [ min-matching(box→storage) + robot→push-square ].
    Dead-locks tested:
        • box in a 2-wall corner (not storage)
        • box against a wall with no storage downstream
        • frozen 2×2 block of boxes (none on storage)
    """
    boxes   = [b for b in state.boxes   if b not in state.storage]
    stores  = [s for s in state.storage if s not in state.boxes]
    if not boxes:
        return 0

    W, H      = state.width, state.height
    obstacles = state.obstacles

    # ---------- quick dead-lock screens ----------
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

    # ---------- minimum-matching cost ----------
    m = len(boxes)
    if len(stores) < m:
        return math.inf
    match_cost = _min_matching_cost(boxes, stores[:m])

    # ---------- robot → closest valid push-square ----------
    robot_push = min(_robot_to_push_sq(state, b) for b in boxes)
    if robot_push == math.inf:
        return math.inf

    return match_cost + robot_push



def heur_zero(state):
    '''Zero Heuristic can be used to make A* search perform uniform cost search'''
    return 0

def heur_manhattan_distance(state):
    # IMPLEMENT
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
    # IMPLEMENT
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
    # IMPLEMENT    
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
    start_time = os.times()[0]
    best_state = False
    best_cost = float('inf')
    final_stats = None

    # Instead of decrementing by 1, use a multiplier to reduce weight more gradually
    current_weight = weight
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
            if sol_state.gval < best_cost:
                best_cost = sol_state.gval
                best_state = sol_state
                final_stats = stats
            
            # Reduce weight more gradually
            current_weight = max(1, current_weight * 0.8)
        else:
            # If no solution found, reduce weight less aggressively
            current_weight = max(1, current_weight * 0.9)
            
        if current_weight == 1:
            break

    return best_state, final_stats




def iterative_gbfs(initial_state, heur_fn, timebound=5):
    '''Enhanced iterative GBFS with dynamic time slicing and pruning'''
    start_time = os.times()[0]
    best_state = False
    best_cost = float('inf')
    final_stats = None
    
    # Keep track of visited states to avoid repeated work
    visited_costs = {}  # state hash -> cost
    consecutive_failures = 0
    min_time_slice = 0.1  # Minimum time slice to attempt
    
    while True:
        now = os.times()[0]
        remaining = timebound - (now - start_time)
        if remaining <= 0:
            break
            
        # Dynamic time slice calculation
        if best_state:
            # If we have a solution, use shorter time slices
            time_slice = min(0.1 * remaining, remaining)
        else:
            # If no solution yet, use longer time slices
            time_slice = min(0.25 * remaining, remaining)
            
        # Don't bother with very short time slices
        if time_slice < min_time_slice:
            break
            
        se = SearchEngine('best_first')
        se.init_search(initial_state, sokoban_goal_state, heur_fn)
        
        # Enhanced costbound with both g and h bounds
        costbound = (best_cost, best_cost * 1.5, float('inf'))
        
        sol_state, stats = se.search(time_slice, costbound)
        
        if not sol_state:
            consecutive_failures += 1
            # If we've failed multiple times, maybe increase min_time_slice
            if consecutive_failures > 3:
                min_time_slice *= 1.5
            if consecutive_failures > 5:
                break  # Give up after too many consecutive failures
            continue
            
        consecutive_failures = 0  # Reset failure counter
        
        # Check if this is a new best solution
        if sol_state.gval < best_cost:
            state_hash = sol_state.hashable_state()
            if state_hash not in visited_costs or sol_state.gval < visited_costs[state_hash]:
                best_cost = sol_state.gval
                best_state = sol_state
                final_stats = stats
                visited_costs[state_hash] = sol_state.gval
                # Reduce min_time_slice since we're making progress
                min_time_slice = max(0.1, min_time_slice * 0.9)
        else:
            # If we found a solution but it's not better, maybe we need longer time slices
            min_time_slice = min(0.5, min_time_slice * 1.2)
            if len(visited_costs) > 1000:  # Arbitrary limit
                break  # Give up if we've explored too many states
                
    return best_state, final_stats



