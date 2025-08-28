import time
from puzzle_csp import binary_ne_grid, nary_ad_grid, caged_csp
from propagators import prop_BT, prop_FC, prop_GAC
from cspbase import BT

# Example FunPuzz boards (domain ≤ 6)
BOARDS = [
    # 3x3 example (valid FunPuzz format)
    [
        [3],
        [11, 12, 13, 6, 0],      # Row 1: 1+2+3=6
        [21, 22, 23, 6, 0],      # Row 2: 1+2+3=6
        [31, 32, 33, 6, 0],      # Row 3: 1+2+3=6
        [11, 21, 31, 6, 0],      # Col 1: 1+2+3=6
        [12, 22, 32, 6, 0],      # Col 2: 1+2+3=6
        [13, 23, 33, 6, 0]       # Col 3: 1+2+3=6
    ],
    # 4x4 example (valid FunPuzz format)
    [
        [4],
        [11, 12, 13, 14, 10, 0],    # Row 1: 1+2+3+4=10
        [21, 22, 23, 24, 10, 0],    # Row 2: 1+2+3+4=10
        [31, 32, 33, 34, 10, 0],    # Row 3: 1+2+3+4=10
        [41, 42, 43, 44, 10, 0],    # Row 4: 1+2+3+4=10
        [11, 21, 31, 41, 10, 0],    # Col 1: 1+2+3+4=10
        [12, 22, 32, 42, 10, 0],    # Col 2: 1+2+3+4=10
        [13, 23, 33, 43, 10, 0],    # Col 3: 1+2+3+4=10
        [14, 24, 34, 44, 10, 0]     # Col 4: 1+2+3+4=10
    ],
    # Add more boards as needed, up to 6x6
]

ENCODINGS = [
    ("binary_ne_grid", binary_ne_grid),
    ("nary_ad_grid", nary_ad_grid),
    ("caged_csp", caged_csp)
]

PROPAGATORS = [
    ("prop_BT", prop_BT),
    ("prop_FC", prop_FC),
    ("prop_GAC", prop_GAC)
]

TIMEOUT = 120  # seconds

def run_test(board, encoding_name, encoding_func, propagator_name, propagator_func):
    print(f"Testing {encoding_name} with {propagator_name} on board size {board[0][0]}")
    start = time.time()
    csp, var_array = encoding_func(board)
    solver = BT(csp)
    try:
        solver.bt_search(propagator_func)
    except Exception as e:
        print(f"Error: {e}")
    elapsed = time.time() - start
    print(f"Time taken: {elapsed:.7f} seconds")
    if elapsed > TIMEOUT:
        print("FAILED: Exceeded 120 seconds!")
    else:
        print("PASSED: Within time limit.")
    print("-" * 40)

if __name__ == "__main__":
    for board in BOARDS:
        for encoding_name, encoding_func in ENCODINGS:
            for propagator_name, propagator_func in PROPAGATORS:
                run_test(board, encoding_name, encoding_func, propagator_name, propagator_func)