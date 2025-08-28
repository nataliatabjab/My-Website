#!/usr/bin/env python3
from tests import *

import io
import traceback
import argparse
import contextlib
import signal

try:
    import agent
except ImportError:
    pass

#######################################
# UTILITIES & TIMEOUTS
#######################################
class TimeoutException(Exception):
    """Raised when time is up."""
    pass

def _timeout_handler(signum, frame):
    raise TimeoutException("Timeout occurred")

def set_timeout(seconds):
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(seconds)

def reset_timeout():
    """Disable alarm."""
    signal.alarm(0)

def contains_list(lst):
    return any(isinstance(e, list) for e in lst)

def sort_innermost_lists(lst):
    """
    Sort the innermost lists in a list-of-lists-of-lists recursively.
    Used for comparing nested lists ignoring order in the innermost layer.
    """
    if not isinstance(lst, list):
        return
    elif contains_list(lst):
        for e in lst:
            sort_innermost_lists(e)
    else:
        lst.sort()

def log(msg, verbose):
    if verbose:
        print(msg)


TIMEOUT = 60

def main():
    parser = argparse.ArgumentParser(description="Run CSP Futoshiki autograder.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--test", "-t", nargs="+",
                        help="Specify one or more test names to run (e.g. test_simple_fc test_tiny_adder_fc). "
                             "If omitted, all tests will be run.")
    args = parser.parse_args()
    verbose = args.verbose

    print("Running Othello Autograder...\n")
        
    # Helper function: run an individual test.
    # If verbose is not set, redirect output (stdout) to an in-memory buffer.
    def run_test(test_func, *test_args, test_name=""):
        try:
            with contextlib.redirect_stdout(io.StringIO()) if not verbose else contextlib.nullcontext():
                set_timeout(TIMEOUT)  # 20s timeout per test
                s, detail, ms = test_func(*test_args)
                reset_timeout()
            return s, detail, ms
        except TimeoutException:
            return 0, f"{test_name} - TIMEOUT", 1
        except Exception:
            tb = traceback.format_exc()
            return 0, f"{test_name} - RUNTIME ERROR:\n{tb}", 1

    # List of tests including an extra field for the test group
    tests = [
        (compute_utility_test, agent.compute_utility, "Minimax Compute Utility"),
        (select_move_minimax_test, agent.select_move_minimax, "Minimax Select Move"),
        (minimax_min_node_1_test, agent.minimax_min_node, "Minimax Min Node Test (Player 1)"),
        (minimax_max_node_1_test, agent.minimax_max_node, "Minimax Max Node Test (Player 1)"),
        (minimax_min_node_2_test, agent.minimax_min_node, "Minimax Min Node Test (Player 2)"),
        (minimax_max_node_2_test, agent.minimax_max_node, "Minimax Max Node Test (Player 2)"),
        (select_move_alphabeta_test, agent.select_move_alphabeta, "Alpha-Beta Select Move"),
        (select_move_equal_test, (agent.select_move_minimax, agent.select_move_alphabeta), "Alpha-Beta Equal Select Move"),
        (alphabeta_min_node_1_test, agent.alphabeta_min_node, "Alpha-Beta Min Node Test (Player 1)"),
        (alphabeta_max_node_1_test, agent.alphabeta_max_node, "Alpha-Beta Max Node Test (Player 1)"),
        (alphabeta_min_node_2_test, agent.alphabeta_min_node, "Alpha-Beta Min Node Test (Player 2)"),
        (alphabeta_max_node_2_test, agent.alphabeta_max_node, "Alpha-Beta Max Node Test (Player 2)"),
        (ordering_small_test, agent.select_move_alphabeta, "Ordering (Small Boards)"),
        (ordering_big_test, agent.select_move_alphabeta, "Ordering (Big Boards)"),
        (caching_small_test, agent.select_move_alphabeta, "Caching (Small Boards)"),
        (caching_big_test, agent.select_move_alphabeta, "Caching (Big Boards)"),
    ]

    # If the user provided specific test names, filter out tests not matching those names.
    if args.test:
        specified = set(args.test)
        tests = [t for t in tests if t[2] in specified]
        if not tests:
            print("No matching tests found for the provided names. Exiting.")
            return

    # Initialize dictionaries to track scores per group.
    overall_score = 0
    overall_max = 0

    # Run each test, and print a formatted result.
    for test_func, test_arg, test_name in tests:
        s, detail, ms = run_test(test_func, test_arg, test_name=test_name)
        overall_score += s
        overall_max += ms

        # Determine status tag based on score
        if s == ms:
            status = "[PASSED]"
        elif s > 0:
            status = "[PARTIAL]"
        else:
            status = "[FAIL]"

        # If no details, print "None"
        detail_to_print = detail.strip() if detail.strip() else "None"

        # Print the test result in the desired format.
        print(f"{status} {test_name} => score: {s}/{ms} details: {detail_to_print}")

    print("Overall Test Score: %d/%d" % (overall_score, overall_max))


if __name__ == "__main__":
    main()
