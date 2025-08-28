from puzzle_csp import caged_csp
from propagators import prop_GAC, prop_FC
from cspbase import BT


def pretty(grid):
    for row in grid:
        print(*[v.get_assigned_value() for v in row])
    print()


def solve(board, propagator, tag):
    print(f"\n=== {tag} ===")
    csp, g = caged_csp(board)
    BT(csp).bt_search(propagator)
    pretty(g)


if __name__ == "__main__":
    board = [
        [6],
        [11, 21, 11, 0],
        [12, 13, 2, 2],
        [22, 23, 3, 1],
        [14, 24, 20, 3],
        [15, 16, 26, 36, 6, 3],
        [31, 41, 32, 42, 240, 3],
        [33, 34, 6, 3],
        [25, 35, 3, 2],
        [51, 52, 6, 3],
        [53, 43, 6, 3],
        [54, 44, 55, 7, 0],
        [45, 46, 30, 3],
        [56, 66, 9, 0],
        [61, 62, 63, 8, 0],
        [64, 65, 2, 2]
    ]

    solve(board, prop_GAC, "prop_GAC")
    solve(board, prop_FC,  "prop_FC")
