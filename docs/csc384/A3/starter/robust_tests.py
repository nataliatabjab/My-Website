"""
robust_tests.py
================

This module defines a suite of additional tests for the `agent.py` Othello AI
to supplement the provided starter tests.  The goal of these tests is to
exercise the search and evaluation routines on a wider variety of board
configurations than the basic 4x4 examples.  The tests cover terminal
positions, situations in which the current player has no legal move,
single–move positions, larger boards and deeper search limits, the
effects of enabling move ordering, and basic sanity checks on the
heuristic evaluation function.

To run all of the tests simply execute this module directly.  Each test
prints a short message on success and raises an AssertionError with a
descriptive message if a check fails.

The tests rely only on the public functions exported by `agent.py` and
`othello_shared.py`.  There are no dependencies on the original test
driver, so this file can serve as a standalone confidence suite for
validating your Othello AI.
"""

import os
import time

import agent
from othello_shared import get_possible_moves, get_score


###############################################################################
# Board definitions
###############################################################################

# A completely filled 4×4 board.  Player 1 controls nine discs and player 2
# controls seven, so the utility for player 1 is +2 and for player 2 is –2.
full_board_4x4 = (
    (1, 1, 1, 1),
    (1, 1, 2, 2),
    (1, 2, 2, 2),
    (1, 2, 1, 2),
)

# A 6×6 position near the end of the game.  Both players have no legal
# moves and the final score difference is −12 for player 1 (so the
# utility for colour 1 is −12 and for colour 2 is +12).
big_endgame_6x6 = (
    (2, 2, 2, 2, 2, 2),
    (2, 2, 1, 1, 1, 2),
    (2, 1, 1, 1, 1, 2),
    (2, 1, 1, 1, 2, 2),
    (2, 1, 1, 2, 2, 2),
    (2, 2, 2, 2, 2, 2),
)

# Positions where the current player has no legal move but the opponent
# does.  In the first board player 1 has no move and player 2 has
# several; in the second the roles are reversed.  In both cases the
# non‐mover should return `None` while the opponent still has a valid
# move available.
no_move_board_c1 = (
    (2, 2, 2, 2),
    (2, 2, 2, 2),
    (1, 1, 1, 0),
    (0, 0, 0, 0),
)

no_move_board_c2 = (
    (1, 1, 1, 1),
    (1, 1, 1, 1),
    (2, 2, 2, 0),
    (0, 0, 0, 0),
)

# Boards in which exactly one legal move exists for the current player.
# There is a 4×4 board with one move for player 1 and its colour–swapped
# counterpart for player 2, and similar 6×6 boards.
single_move_board4_c1 = (
    (2, 1, 2, 2),
    (1, 2, 0, 0),
    (2, 2, 1, 2),
    (1, 1, 1, 2),
)
single_move_board4_c2 = (
    (1, 2, 1, 1),
    (2, 1, 0, 0),
    (1, 1, 2, 1),
    (2, 2, 2, 1),
)

single_move_board6_c1 = (
    (2, 1, 2, 2, 2, 2),
    (1, 2, 0, 0, 2, 2),
    (2, 2, 1, 2, 2, 2),
    (1, 1, 1, 2, 2, 2),
    (2, 2, 2, 2, 2, 2),
    (2, 2, 2, 2, 2, 2),
)
single_move_board6_c2 = (
    (1, 2, 1, 1, 1, 1),
    (2, 1, 0, 0, 1, 1),
    (1, 1, 2, 1, 1, 1),
    (2, 2, 2, 1, 1, 1),
    (1, 1, 1, 1, 1, 1),
    (1, 1, 1, 1, 1, 1),
)

# An 8×8 mid–game board used to test deeper searches.  Both players have
# multiple moves available.  When searching to a depth of 3, both
# minimax and alpha–beta agree on the same move for each colour.
large_board_8x8 = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 2, 2, 2, 2, 2, 2, 0),
    (0, 2, 1, 1, 1, 1, 2, 0),
    (0, 2, 1, 0, 0, 1, 2, 0),
    (0, 2, 1, 0, 0, 1, 2, 0),
    (0, 2, 1, 1, 1, 1, 2, 0),
    (0, 2, 2, 2, 2, 2, 2, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
)

# A 6×6 board where enabling move ordering results in a different best
# move than the default exploration order.  This board was found by
# random search: without ordering the best move for player 1 at depth
# 2 is (1, 3) but with ordering enabled the chosen move becomes (2, 5).
ordering_board_6x6 = (
    (2, 0, 2, 0, 1, 1),
    (2, 0, 1, 0, 2, 0),
    (1, 1, 0, 1, 2, 2),
    (0, 0, 2, 2, 1, 0),
    (2, 1, 2, 2, 2, 1),
    (2, 2, 0, 1, 1, 2),
)

# A simple 4×4 board with multiple legal moves used for testing that
# enabling ordering does not affect the move returned on small boards.
ordering_small_board_4x4 = (
    (0, 0, 0, 0),
    (0, 2, 1, 0),
    (0, 1, 1, 1),
    (0, 0, 0, 0),
)

# Boards used to check the heuristic evaluation.  In the first board
# player 1 occupies all of the edges and corners on a 6×6 board while
# player 2 holds the interior squares; in the second board the roles
# are reversed.  The heuristic should strongly favour the player with
# the stable edge control.
heuristic_board_c1_edges = (
    (1, 1, 1, 1, 1, 1),
    (1, 2, 2, 2, 2, 1),
    (1, 2, 0, 0, 2, 1),
    (1, 2, 0, 0, 2, 1),
    (1, 2, 2, 2, 2, 1),
    (1, 1, 1, 1, 1, 1),
)
heuristic_board_c2_edges = tuple(
    tuple(2 if x == 1 else 1 if x == 2 else 0 for x in row)
    for row in heuristic_board_c1_edges
)


###############################################################################
# Helper functions for assertions and timing
###############################################################################

def assert_equal(val1, val2, msg=""):
    """Raise an AssertionError with a helpful message if val1 != val2."""
    if val1 != val2:
        raise AssertionError(f"Expected {val2}, got {val1}. {msg}")


def assert_is_none(val, msg=""):
    """Raise an AssertionError if val is not None."""
    if val is not None:
        raise AssertionError(f"Expected None, got {val}. {msg}")


def assert_true(condition, msg=""):
    """Raise an AssertionError if condition is false."""
    if not condition:
        raise AssertionError(msg)


###############################################################################
# Test suites
###############################################################################

def test_terminal_positions():
    """Check that terminal positions are handled correctly."""
    # Test the 4×4 full board
    p1_count, p2_count = get_score(full_board_4x4)
    # Utility values
    util_1 = agent.compute_utility(full_board_4x4, 1)
    util_2 = agent.compute_utility(full_board_4x4, 2)
    # For this board player 1 leads by 2 discs
    assert_equal(util_1, p1_count - p2_count,
                 "compute_utility should return score difference for colour 1")
    assert_equal(util_2, p2_count - p1_count,
                 "compute_utility should return score difference for colour 2")
    # No legal moves for either colour
    assert_equal(get_possible_moves(full_board_4x4, 1), [],
                 "full board should have no moves for player 1")
    assert_equal(get_possible_moves(full_board_4x4, 2), [],
                 "full board should have no moves for player 2")
    # Minimax and alpha–beta should return None regardless of depth
    assert_is_none(agent.select_move_minimax(full_board_4x4, 1, 3),
                   "select_move_minimax should return None on terminal boards")
    assert_is_none(agent.select_move_minimax(full_board_4x4, 2, 3),
                   "select_move_minimax should return None on terminal boards")
    assert_is_none(agent.select_move_alphabeta(full_board_4x4, 1, 3),
                   "select_move_alphabeta should return None on terminal boards")
    assert_is_none(agent.select_move_alphabeta(full_board_4x4, 2, 3),
                   "select_move_alphabeta should return None on terminal boards")

    # Test the 6×6 end–game board
    p1_count, p2_count = get_score(big_endgame_6x6)
    util_1 = agent.compute_utility(big_endgame_6x6, 1)
    util_2 = agent.compute_utility(big_endgame_6x6, 2)
    # Utility should equal the score difference
    assert_equal(util_1, p1_count - p2_count,
                 "utility on terminal boards should reflect final score for colour 1")
    assert_equal(util_2, p2_count - p1_count,
                 "utility on terminal boards should reflect final score for colour 2")
    # Both players should have no moves
    assert_equal(get_possible_moves(big_endgame_6x6, 1), [],
                 "end–game board should have no moves for player 1")
    assert_equal(get_possible_moves(big_endgame_6x6, 2), [],
                 "end–game board should have no moves for player 2")
    # Search functions should return None
    assert_is_none(agent.select_move_minimax(big_endgame_6x6, 1, 3),
                   "select_move_minimax should return None on terminal boards")
    assert_is_none(agent.select_move_minimax(big_endgame_6x6, 2, 3),
                   "select_move_minimax should return None on terminal boards")
    assert_is_none(agent.select_move_alphabeta(big_endgame_6x6, 1, 3),
                   "select_move_alphabeta should return None on terminal boards")
    assert_is_none(agent.select_move_alphabeta(big_endgame_6x6, 2, 3),
                   "select_move_alphabeta should return None on terminal boards")
    print("test_terminal_positions passed")


def test_no_move_positions():
    """Check that positions with no legal moves for the current player are
    handled properly (the search should return None)."""
    # Player 1 has no moves on no_move_board_c1
    assert_equal(get_possible_moves(no_move_board_c1, 1), [],
                 "player 1 should have no moves on no_move_board_c1")
    assert_true(get_possible_moves(no_move_board_c1, 2) != [],
                "player 2 should have at least one move on no_move_board_c1")
    assert_is_none(agent.select_move_minimax(no_move_board_c1, 1, 4),
                   "minimax should return None when no moves are available")
    assert_is_none(agent.select_move_alphabeta(no_move_board_c1, 1, 4),
                   "alpha–beta should return None when no moves are available")
    # Player 2 still has a valid move
    expected_move = (0, 3)
    assert_equal(agent.select_move_minimax(no_move_board_c1, 2, 4), expected_move,
                 "minimax should choose the only available move for player 2 on no_move_board_c1")
    assert_equal(agent.select_move_alphabeta(no_move_board_c1, 2, 4), expected_move,
                 "alpha–beta should choose the only available move for player 2 on no_move_board_c1")

    # Mirror situation: player 2 has no moves on no_move_board_c2
    assert_equal(get_possible_moves(no_move_board_c2, 2), [],
                 "player 2 should have no moves on no_move_board_c2")
    assert_true(get_possible_moves(no_move_board_c2, 1) != [],
                "player 1 should have at least one move on no_move_board_c2")
    assert_is_none(agent.select_move_minimax(no_move_board_c2, 2, 4),
                   "minimax should return None when no moves are available")
    assert_is_none(agent.select_move_alphabeta(no_move_board_c2, 2, 4),
                   "alpha–beta should return None when no moves are available")
    # Player 1 should pick the bottom–left move
    expected_move = (0, 3)
    assert_equal(agent.select_move_minimax(no_move_board_c2, 1, 4), expected_move,
                 "minimax should choose the only available move for player 1 on no_move_board_c2")
    assert_equal(agent.select_move_alphabeta(no_move_board_c2, 1, 4), expected_move,
                 "alpha–beta should choose the only available move for player 1 on no_move_board_c2")
    print("test_no_move_positions passed")


def test_single_move_positions():
    """Ensure that positions with exactly one legal move return that move."""
    # 4×4, player 1
    moves = get_possible_moves(single_move_board4_c1, 1)
    assert_equal(len(moves), 1,
                 "single_move_board4_c1 should have exactly one move for player 1")
    expected = moves[0]
    assert_equal(agent.select_move_minimax(single_move_board4_c1, 1, 4), expected,
                 "minimax should return the single legal move on single_move_board4_c1")
    assert_equal(agent.select_move_alphabeta(single_move_board4_c1, 1, 4), expected,
                 "alpha–beta should return the single legal move on single_move_board4_c1")
    # 4×4, player 2
    moves = get_possible_moves(single_move_board4_c2, 2)
    assert_equal(len(moves), 1,
                 "single_move_board4_c2 should have exactly one move for player 2")
    expected = moves[0]
    assert_equal(agent.select_move_minimax(single_move_board4_c2, 2, 4), expected,
                 "minimax should return the single legal move on single_move_board4_c2")
    assert_equal(agent.select_move_alphabeta(single_move_board4_c2, 2, 4), expected,
                 "alpha–beta should return the single legal move on single_move_board4_c2")
    # 6×6, player 1
    moves = get_possible_moves(single_move_board6_c1, 1)
    assert_equal(len(moves), 1,
                 "single_move_board6_c1 should have exactly one move for player 1")
    expected = moves[0]
    assert_equal(agent.select_move_minimax(single_move_board6_c1, 1, 3), expected,
                 "minimax should return the single legal move on single_move_board6_c1")
    assert_equal(agent.select_move_alphabeta(single_move_board6_c1, 1, 3), expected,
                 "alpha–beta should return the single legal move on single_move_board6_c1")
    # 6×6, player 2
    moves = get_possible_moves(single_move_board6_c2, 2)
    assert_equal(len(moves), 1,
                 "single_move_board6_c2 should have exactly one move for player 2")
    expected = moves[0]
    assert_equal(agent.select_move_minimax(single_move_board6_c2, 2, 3), expected,
                 "minimax should return the single legal move on single_move_board6_c2")
    assert_equal(agent.select_move_alphabeta(single_move_board6_c2, 2, 3), expected,
                 "alpha–beta should return the single legal move on single_move_board6_c2")
    print("test_single_move_positions passed")


def test_large_board_positions():
    """Check that minimax and alpha–beta agree on larger boards and deeper
    search depths."""
    # Both players have moves on the 8×8 board.  Search to depth 3.
    # Player 1 move
    mm_move1 = agent.select_move_minimax(large_board_8x8, 1, 3)
    ab_move1 = agent.select_move_alphabeta(large_board_8x8, 1, 3)
    assert_equal(mm_move1, ab_move1,
                 "minimax and alpha–beta should agree on the move for player 1 on large_board_8x8")
    # Player 2 move
    mm_move2 = agent.select_move_minimax(large_board_8x8, 2, 3)
    ab_move2 = agent.select_move_alphabeta(large_board_8x8, 2, 3)
    assert_equal(mm_move2, ab_move2,
                 "minimax and alpha–beta should agree on the move for player 2 on large_board_8x8")
    # For the chosen board and depth we expect the following best moves
    assert_equal(mm_move1, (0, 2),
                 "unexpected best move for player 1 on large_board_8x8")
    assert_equal(mm_move2, (3, 3),
                 "unexpected best move for player 2 on large_board_8x8")
    print("test_large_board_positions passed")


def test_ordering_flag():
    """Verify that enabling move ordering behaves sensibly.

    On small (4×4) boards the ordering flag should have no effect on the
    chosen move.  On a carefully constructed 6×6 board the choice
    changes depending on whether ordering is enabled.
    """
    # Small board: ordering should not affect the result
    move_no_order = agent.select_move_alphabeta(ordering_small_board_4x4, 1, 4, 0, 0)
    move_with_order = agent.select_move_alphabeta(ordering_small_board_4x4, 1, 4, 0, 1)
    assert_equal(move_no_order, move_with_order,
                 "ordering should not change the move on small boards")

    # Large board: ordering can change the selected move.  We use the
    # lower–level alphabeta_max_node to avoid the busy loop in
    # select_move_alphabeta (which intentionally consumes CPU time when
    # ordering is disabled).  The behaviour of alphabeta_max_node with
    # ordering=0 versus ordering=1 should differ on this board.
    move_no_order, val_no_order = agent.alphabeta_max_node(
        ordering_board_6x6, 1, float('-inf'), float('inf'), 2, 0, 0
    )
    move_with_order, val_with_order = agent.alphabeta_max_node(
        ordering_board_6x6, 1, float('-inf'), float('inf'), 2, 0, 1
    )
    # Assert that the moves differ
    assert_true(move_no_order != move_with_order,
                "ordering should change the chosen move on the ordering_board_6x6")
    # Check against the expected moves discovered during test construction
    assert_equal(move_no_order, (1, 3),
                 "unexpected move for ordering=0 on ordering_board_6x6")
    assert_equal(move_with_order, (2, 5),
                 "unexpected move for ordering=1 on ordering_board_6x6")
    print("test_ordering_flag passed")


def test_heuristic_sanity():
    """Perform a basic sanity check on the heuristic evaluation.

    The heuristic should reward positions where the player controls
    corners and edges, and penalise positions where the opponent holds
    those stable squares.  We test symmetric boards and check the sign
    of the heuristic values.
    """
    val_c1 = agent.compute_heuristic(heuristic_board_c1_edges, 1)
    val_c1_opponent = agent.compute_heuristic(heuristic_board_c1_edges, 2)
    # On heuristic_board_c1_edges player 1 controls all corners and edges,
    # so the evaluation should be strongly positive for player 1 and
    # strongly negative for player 2.
    assert_true(val_c1 > 0,
                "heuristic should be positive when player 1 holds the corners and edges")
    assert_true(val_c1_opponent < 0,
                "heuristic should be negative for the opponent when player 1 holds the edges")
    # On the colour–swapped board the signs should be reversed
    val_c2 = agent.compute_heuristic(heuristic_board_c2_edges, 2)
    val_c2_opponent = agent.compute_heuristic(heuristic_board_c2_edges, 1)
    assert_true(val_c2 > 0,
                "heuristic should be positive when player 2 holds the corners and edges")
    assert_true(val_c2_opponent < 0,
                "heuristic should be negative for the opponent when player 2 holds the edges")
    # The magnitude for opposite colours on the same board should match
    assert_equal(val_c1, -val_c1_opponent,
                 "heuristic should be symmetric for opposite colours")
    assert_equal(val_c2, -val_c2_opponent,
                 "heuristic should be symmetric for opposite colours")
    print("test_heuristic_sanity passed")


def test_caching_no_change():
    """Ensure that enabling state caching does not change the chosen move.

    We run alpha–beta on a moderately complex board at a modest depth with
    and without caching and verify that the selected moves are identical.
    """
    # Use the large 8×8 board for this test.  Caching should never
    # change the result of the search.
    move_no_cache_1 = agent.select_move_alphabeta(large_board_8x8, 1, 4, 0, 0)
    move_cache_1 = agent.select_move_alphabeta(large_board_8x8, 1, 4, 1, 0)
    assert_equal(move_no_cache_1, move_cache_1,
                 "caching should not change the selected move for player 1")
    move_no_cache_2 = agent.select_move_alphabeta(large_board_8x8, 2, 4, 0, 0)
    move_cache_2 = agent.select_move_alphabeta(large_board_8x8, 2, 4, 1, 0)
    assert_equal(move_no_cache_2, move_cache_2,
                 "caching should not change the selected move for player 2")
    print("test_caching_no_change passed")


###############################################################################
# Test runner
###############################################################################

def run_all_tests():
    """Execute all defined tests in sequence."""
    test_terminal_positions()
    test_no_move_positions()
    test_single_move_positions()
    test_large_board_positions()
    test_ordering_flag()
    test_heuristic_sanity()
    test_caching_no_change()
    print("\nAll robust tests passed successfully.")


if __name__ == "__main__":
    run_all_tests()