"""
Heuristic testing script for Othello agents.

Plays minimax games between a utility-based agent and a heuristic-based
agent over a range of depths.  The total score of the heuristic agent
is printed (1 = win, 0.5 = draw).

Run, e.g.:
    python heuristic_test.py --heuristic improved --min-depth 2 --max-depth 6
"""

import argparse
from typing import Callable

from othello_shared import get_possible_moves, play_move, get_score
import agent   # your existing agent.py


# --------------------------------------------------------------------------- #
# Generic minimax helper (no caching / pruning—good enough for testing).      #
# --------------------------------------------------------------------------- #
def minimax_move(board, color, depth, eval_fn: Callable[[tuple, int], int]):
    legal = get_possible_moves(board, color)
    if depth == 0 or not legal:
        return None, eval_fn(board, color)

    best_move = None
    if color == 1:                           # max
        best_val = float('-inf')
        for mv in legal:
            new_b = play_move(board, color, *mv)
            _, val = minimax_move(new_b, 2, depth - 1, eval_fn)
            if val > best_val:
                best_val, best_move = val, mv
    else:                                    # min
        best_val = float('inf')
        for mv in legal:
            new_b = play_move(board, color, *mv)
            _, val = minimax_move(new_b, 1, depth - 1, eval_fn)
            if val < best_val:
                best_val, best_move = val, mv
    return best_move, best_val


# --------------------------------------------------------------------------- #
# Play a single game with one side using a chosen heuristic.                  #
# --------------------------------------------------------------------------- #
def play_game(depth: int, heuristic_colour: int,
              heuristic_fn: Callable[[tuple, int], int]) -> float:
    board = (
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 1, 2, 0, 0, 0),
        (0, 0, 0, 2, 1, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
    )

    def eval_fn(state, root):
        return (heuristic_fn if root == heuristic_colour else agent.compute_utility)(state, root)

    player = 1
    while True:
        moves = get_possible_moves(board, player)
        if moves:
            mv, _ = minimax_move(board, player, depth, eval_fn)
            if mv:
                board = play_move(board, player, *mv)

        opp = 3 - player
        if not moves and not get_possible_moves(board, opp):
            break
        player = opp

    p1, p2 = get_score(board)
    my, opp = (p1, p2) if heuristic_colour == 1 else (p2, p1)
    return 1.0 if my > opp else 0.5 if my == opp else 0.0


# --------------------------------------------------------------------------- #
# Baseline heuristics from earlier code                                       #
# --------------------------------------------------------------------------- #
def weighted(board, color):
    return agent.compute_heuristic(board, color)


def mobility(board, color):
    return len(get_possible_moves(board, color)) - len(get_possible_moves(board, 3 - color))


def disk_corners(board, color):
    p1, p2 = get_score(board)
    diff = p1 - p2 if color == 1 else p2 - p1
    n = len(board)
    corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
    bonus = 0
    for r, c in corners:
        if board[r][c] == color:
            bonus += 25
        elif board[r][c] != 0:
            bonus -= 25
    return diff + bonus


def weighted_mobility(board, color):
    return agent.compute_heuristic(board, color) + 5 * mobility(board, color)


def corners_weighted(board, color):
    return disk_corners(board, color) + agent.compute_heuristic(board, color)


# --------------------------------------------------------------------------- #
# NEW dynamic “improved” heuristic                                            #
# --------------------------------------------------------------------------- #
# helper functions (underscored = internal only)
def _corner_diff(board, color):
    n = len(board)
    corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
    score = 0
    for r, c in corners:
        if board[r][c] == color:
            score += 1
        elif board[r][c] != 0:
            score -= 1
    return score  # −4 … 4


def _potential_mobility(board, color):
    n = len(board)
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1))

    def adj_empty_to(colour):
        count = 0
        for r in range(n):
            for c in range(n):
                if board[r][c] != 0:
                    continue
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n and board[nr][nc] == colour:
                        count += 1
                        break
        return count

    return adj_empty_to(color) - adj_empty_to(3 - color)


def _stable_discs(board, color):
    n = len(board)
    stable = 0
    for (edge, step) in [((0, 0), (0, 1)), ((0, 0), (1, 0)),
                         ((n - 1, 0), (0, 1)), ((0, n - 1), (1, 0))]:
        r, c = edge
        dr, dc = step
        if board[r][c] == 0:
            continue
        owner = board[r][c]
        while 0 <= r < n and 0 <= c < n and board[r][c] == owner:
            if owner == color:
                stable += 1
            r += dr
            c += dc
    return stable


def improved_weighted_mobility(board, color):
    base = agent.compute_heuristic(board, color)

    empties = sum(row.count(0) for row in board)
    phase = empties / (len(board) ** 2)          # 1 → opening, 0 → endgame

    mob          = mobility(board, color)
    pot_mob      = _potential_mobility(board, color)
    stable       = _stable_discs(board, color)
    corner_net   = _corner_diff(board, color)

    value = (base
             + (10 * phase)        * mob          # diminishes
             + (-5)                * pot_mob      # constant penalty
             + (40 * (1 - phase))  * stable       # grows
             + 100                 * corner_net)  # always large
    return int(value)


# --------------------------------------------------------------------------- #
# Heuristic registry                                                          #
# --------------------------------------------------------------------------- #
HEURISTICS = {
    "weighted":           weighted,
    "mobility":           mobility,
    "disk_corners":       disk_corners,
    "weighted_mobility":  weighted_mobility,
    "corners_weighted":   corners_weighted,
    "improved":           improved_weighted_mobility,   # ← new dynamic one
}


# --------------------------------------------------------------------------- #
# CLI driver                                                                  #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Play heuristic vs utility agents on the 8×8 board.")
    parser.add_argument("--heuristic", choices=HEURISTICS.keys(),
                        default="improved")
    parser.add_argument("--min-depth", type=int, default=2)
    parser.add_argument("--max-depth", type=int, default=6)
    args = parser.parse_args()

    h_fn = HEURISTICS[args.heuristic]
    total, games = 0.0, 0
    for d in range(args.min_depth, args.max_depth + 1):
        total += play_game(d, 1, h_fn)  # heuristic as dark
        total += play_game(d, 2, h_fn)  # heuristic as light
        games += 2

    print(f"Heuristic '{args.heuristic}' total score over {games} games: {total}")


if __name__ == "__main__":
    main()
