import os
import time
import agent

TIME_THRESHOLD = 0.15

# boards of size 4
SMALL_BOARDS = [((0, 0, 0, 0), (0, 2, 1, 0), (0, 1, 1, 1), (0, 0, 0, 0)),
                ((0, 1, 0, 0), (0, 1, 1, 0), (0, 1, 2, 1), (0, 0, 0, 2)),
                ((0, 0, 0, 0), (0, 2, 1, 0), (0, 1, 1, 1), (0, 1, 1, 0)),
                ((0, 1, 0, 0), (0, 2, 2, 0), (0, 1, 2, 1), (0, 0, 2, 2)),
                ((1, 0, 0, 2), (1, 1, 2, 0), (1, 1, 1, 1), (1, 2, 2, 2)),
                ((0, 1, 0, 0), (0, 1, 1, 0), (2, 2, 2, 1), (0, 0, 0, 2))]

# boards of size 6
BIG_BOARDS = [((0, 0, 0, 0, 0, 0), (0, 0, 2, 2, 0, 0), (0, 1, 1, 2, 2, 0), (2, 2, 1, 2, 0, 0), (0, 1, 0, 1, 2, 0),
               (0, 0, 0, 0, 0, 0)),
              ((0, 0, 0, 0, 0, 0), (0, 0, 1, 2, 0, 0), (0, 1, 1, 1, 1, 0), (2, 2, 1, 2, 0, 0), (0, 1, 0, 1, 2, 0),
               (0, 0, 0, 0, 0, 0)),
              ((0, 0, 0, 0, 1, 0), (0, 0, 1, 1, 0, 0), (0, 1, 1, 1, 1, 0), (2, 2, 1, 2, 0, 0), (0, 2, 0, 1, 2, 0),
               (0, 0, 2, 2, 1, 0)),
              ((0, 0, 0, 0, 0, 0), (0, 0, 0, 2, 0, 0), (0, 1, 2, 2, 2, 0), (0, 2, 2, 2, 0, 0), (0, 1, 0, 0, 0, 0),
               (0, 0, 0, 0, 0, 0)),
              ((0, 0, 0, 0, 0, 0), (0, 0, 0, 2, 0, 0), (0, 1, 2, 1, 1, 0), (0, 2, 2, 2, 0, 0), (0, 1, 0, 0, 0, 0),
               (0, 0, 0, 0, 0, 0))]


#######################################
# TEST FUNCTIONS
#######################################
def compute_utility_test(compute_utility, name=""):
    correctvalues = [3, 3, 5, -2, 3, 0]
    correct = 0
    details = ""
    for i, board in enumerate(SMALL_BOARDS):
        try:
            value1 = compute_utility(board, 1)
            value2 = compute_utility(board, 2)
        except Exception as e:
            details += f"Board {i}: Exception {e}\n"
            continue
        if value1 == correctvalues[i] and value2 == -correctvalues[i]:
            correct += 1
        else:
            details += (f"Board {i}: expected ({correctvalues[i]}, {-correctvalues[i]}), "
                        f"got ({value1}, {value2})\n")
    max_score = len(correctvalues)
    return correct, details, max_score


def select_move_minimax_test(select_move_minimax, name=""):
    correctmoves_1 = [(0, 0), (2, 3), (0, 0), (3, 0), (3, 1), (0, 3)]
    correctmoves_2 = [(3, 3), (0, 0), (3, 3), (0, 2), (3, 1), (0, 0)]
    correct = 0
    details = ""
    for i, board in enumerate(SMALL_BOARDS):
        try:
            move1 = select_move_minimax(board, 1, 6)
            move2 = select_move_minimax(board, 2, 6)
        except Exception as e:
            details += f"Board {i}: Exception {e}\n"
            continue
        if move1 == correctmoves_1[i] and move2 == correctmoves_2[i]:
            correct += 1
        else:
            details += (f"Board {i}: expected moves {correctmoves_1[i]}, {correctmoves_2[i]}; "
                        f"got {move1}, {move2}\n")
    max_score = len(correctmoves_1)
    return correct, details, max_score


def select_move_alphabeta_test(select_move_alphabeta, name=""):
    correctmoves_1 = [(0, 0), (2, 3), (0, 0), (3, 0), (3, 1), (0, 3)]
    correctmoves_2 = [(3, 3), (0, 0), (3, 3), (0, 2), (3, 1), (0, 0)]
    correct = 0
    details = ""
    for i, board in enumerate(SMALL_BOARDS):
        try:
            move1 = select_move_alphabeta(board, 1, 6)
            move2 = select_move_alphabeta(board, 2, 6)
        except Exception as e:
            details += f"Board {i}: Exception {e}\n"
            continue
        if move1 == correctmoves_1[i] and move2 == correctmoves_2[i]:
            correct += 1
        else:
            details += (f"Board {i}: expected moves {correctmoves_1[i]}, {correctmoves_2[i]}; "
                        f"got {move1}, {move2}\n")
    max_score = len(correctmoves_1)
    return correct, details, max_score

def select_move_equal_test(funcs, name=""):
    # funcs is a tuple: (select_move_minimax, select_move_alphabeta)
    select_move_minimax, select_move_alphabeta = funcs
    correctmoves_1 = [(0, 0), (2, 3), (0, 0), (3, 0), (3, 1)]
    correctmoves_2 = [(3, 3), (0, 0), (3, 3), (0, 2), (3, 1)]
    correct = 0
    details = ""
    for i in range(len(correctmoves_1)):
        board = SMALL_BOARDS[i]
        try:
            move1_minimax = select_move_minimax(board, 1, 6)
            move2_minimax = select_move_minimax(board, 2, 6)
            move1_ab = select_move_alphabeta(board, 1, 6)
            move2_ab = select_move_alphabeta(board, 2, 6)
        except Exception as e:
            details += f"Board {i}: Exception {e}\n"
            continue
        if move1_minimax == move1_ab == correctmoves_1[i] and move2_minimax == move2_ab == correctmoves_2[i]:
            correct += 1
        else:
            details += (f"Board {i}: expected moves {correctmoves_1[i]}, {correctmoves_2[i]}; "
                        f"got minimax: ({move1_minimax}, {move2_minimax}), "
                        f"alphabeta: ({move1_ab}, {move2_ab})\n")
    max_score = len(correctmoves_1)
    return correct, details, max_score

def caching_test(select_move_alphabeta, boards, name=""):
    import os
    check_time = 0
    check_choice = 0
    details = ""
    for i, board in enumerate(boards):
        try:
            if hasattr(agent, 'cache'):
                agent.cache.clear()
            start_time_1 = os.times()[0]
            no_cache = select_move_alphabeta(board, 1, 8)
            end_time_1 = os.times()[0]
            start_time_2 = os.times()[0]
            with_cache = select_move_alphabeta(board, 1, 8, 1)
            end_time_2 = os.times()[0]
        except Exception as e:
            details += f"Board {i}: Exception {e}\n"
            continue
        if (end_time_2 - start_time_2) - (end_time_1 - start_time_1) < TIME_THRESHOLD:
            check_time += 1
        else:
            details += f"Board {i}: caching did not improve time\n"
        if with_cache == no_cache:
            check_choice += 1
        else:
            details += (f"Board {i}: moves differ with caching: no_cache={no_cache}, "
                        f"with_cache={with_cache}\n")
    total = len(boards) * 2
    score = check_time + check_choice
    return score, details, total

def caching_big_test(select_move_alphabeta, name=""):
    return caching_test(
        select_move_alphabeta, BIG_BOARDS, name)

def caching_small_test(select_move_alphabeta, name=""):
    return caching_test(
        select_move_alphabeta, SMALL_BOARDS, name)

def ordering_test(select_move_alphabeta, boards, expected_move_change, name=""):
    check_time = 0
    check_move = 0
    details = ""
    for i, board in enumerate(boards):
        try:
            start_time_1 = os.times()[0]
            no_order = select_move_alphabeta(board, 1, 7, 0, 0)
            end_time_1 = os.times()[0]
            start_time_2 = os.times()[0]
            with_order = select_move_alphabeta(board, 1, 7, 0, 1)
            end_time_2 = os.times()[0]
        except Exception as e:
            details += f"Board {i}: Exception {e}\n"
            continue
        if (end_time_2 - start_time_2) - (end_time_1 - start_time_1) < TIME_THRESHOLD:
            check_time += 1
        else:
            details += f"Board {i}: ordering did not improve time\n"
        if (with_order != no_order) == expected_move_change[i]:
            check_move += 1
        else:
            details += (f"Board {i}: you were {(with_order != no_order)} for ordering change, "
                        f"but expected was {expected_move_change[i]}\n")
    # Don't count the move differences in the score
    total = len(boards) * 2
    score = check_time + check_move
    return score, details, total

def ordering_small_test(select_move_alphabeta, name=""):
    return ordering_test(select_move_alphabeta, SMALL_BOARDS, [False, False, False, False, False, False], name)

def ordering_big_test(select_move_alphabeta, name=""):
    return ordering_test(select_move_alphabeta, BIG_BOARDS, [False, True, False, True, True], name)

def alphabeta_min_node_1_test(alphabeta_min_node, name=""):
    answers = [((2, 4), -10), ((1, 1), -4), ((3, 0), -6), ((0, 1), -8), ((5, 2), -6)]
    correct_moves = 0
    correct_values = 0
    details = ""
    for i, board in enumerate(BIG_BOARDS[:len(answers)]):
        try:
            move, value = alphabeta_min_node(board, 1, float("-Inf"), float("Inf"), 1, 0, 0)
        except Exception as e:
            details += f"Board {i}: Exception {e}\n"
            continue
        expected_move, expected_value = answers[i]
        if move == expected_move:
            correct_moves += 1
        else:
            details += f"Board {i}: expected move {expected_move}, got {move}\n"
        if value == expected_value:
            correct_values += 1
        else:
            details += f"Board {i}: expected value {expected_value}, got {value}\n"
    score = correct_moves + correct_values
    max_score = 2 * len(answers)
    return score, details, max_score


def alphabeta_max_node_1_test(alphabeta_max_node, name=""):
    # Only testing selected boards (indices 1, 2, and 4)
    answers = {1: ((5, 5), 8), 2: ((1, 5), 12), 4: ((3, 4), 4)}
    correct_moves = 0
    correct_values = 0
    details = ""
    for i in sorted(answers.keys()):
        board = BIG_BOARDS[i]
        try:
            move, value = alphabeta_max_node(board, 1, float("-Inf"), float("Inf"), 1, 0, 0)
        except Exception as e:
            details += f"Board {i}: Exception {e}\n"
            continue
        expected_move, expected_value = answers[i]
        if move == expected_move:
            correct_moves += 1
        else:
            details += f"Board {i}: expected move {expected_move}, got {move}\n"
        if value == expected_value:
            correct_values += 1
        else:
            details += f"Board {i}: expected value {expected_value}, got {value}\n"
    score = correct_moves + correct_values
    max_score = 2 * len(answers)
    return score, details, max_score


def minimax_min_node_1_test(minimax_min_node, name=""):
    answers = [((2, 4), -10), ((1, 1), -4), ((3, 0), -6), ((0, 1), -8), ((5, 2), -6)]
    correct_moves = 0
    correct_values = 0
    details = ""
    for i, board in enumerate(BIG_BOARDS[:len(answers)]):
        try:
            move, value = minimax_min_node(board, 1, 1, 0)
        except Exception as e:
            details += f"Board {i}: Exception {e}\n"
            continue
        expected_move, expected_value = answers[i]
        if move == expected_move:
            correct_moves += 1
        else:
            details += f"Board {i}: expected move {expected_move}, got {move}\n"
        if value == expected_value:
            correct_values += 1
        else:
            details += f"Board {i}: expected value {expected_value}, got {value}\n"
    score = correct_moves + correct_values
    max_score = 2 * len(answers)
    return score, details, max_score


def minimax_max_node_1_test(minimax_max_node, name=""):
    # Only testing selected boards (indices 1, 2, and 4)
    answers = {1: ((5, 5), 8), 2: ((1, 5), 12), 4: ((3, 4), 4)}
    correct_moves = 0
    correct_values = 0
    details = ""
    for i in sorted(answers.keys()):
        board = BIG_BOARDS[i]
        try:
            move, value = minimax_max_node(board, 1, 1, 0)
        except Exception as e:
            details += f"Board {i}: Exception {e}\n"
            continue
        expected_move, expected_value = answers[i]
        if move == expected_move:
            correct_moves += 1
        else:
            details += f"Board {i}: expected move {expected_move}, got {move}\n"
        if value == expected_value:
            correct_values += 1
        else:
            details += f"Board {i}: expected value {expected_value}, got {value}\n"
    score = correct_moves + correct_values
    max_score = 2 * len(answers)
    return score, details, max_score


def alphabeta_min_node_2_test(alphabeta_min_node, name=""):
    answers = [((3, 0), -6), ((5, 5), -8), ((1, 5), -12), ((5, 2), -2), ((3, 4), -4)]
    correct_moves = 0
    correct_values = 0
    details = ""
    for i, board in enumerate(BIG_BOARDS[:len(answers)]):
        try:
            move, value = alphabeta_min_node(board, 2, float("-Inf"), float("Inf"), 1, 0, 0)
        except Exception as e:
            details += f"Board {i}: Exception {e}\n"
            continue
        expected_move, expected_value = answers[i]
        if move == expected_move:
            correct_moves += 1
        else:
            details += f"Board {i}: expected move {expected_move}, got {move}\n"
        if value == expected_value:
            correct_values += 1
        else:
            details += f"Board {i}: expected value {expected_value}, got {value}\n"
    score = correct_moves + correct_values
    max_score = 2 * len(answers)
    return score, details, max_score


def alphabeta_max_node_2_test(alphabeta_max_node, name=""):
    # Testing selected boards (indices 1, 2, and 4)
    answers = {1: ((1, 1), 4), 2: ((3, 0), 6), 4: ((5, 2), 6)}
    correct_moves = 0
    correct_values = 0
    details = ""
    for i in sorted(answers.keys()):
        board = BIG_BOARDS[i]
        try:
            move, value = alphabeta_max_node(board, 2, float("-Inf"), float("Inf"), 1, 0, 0)
        except Exception as e:
            details += f"Board {i}: Exception {e}\n"
            continue
        expected_move, expected_value = answers[i]
        if move == expected_move:
            correct_moves += 1
        else:
            details += f"Board {i}: expected move {expected_move}, got {move}\n"
        if value == expected_value:
            correct_values += 1
        else:
            details += f"Board {i}: expected value {expected_value}, got {value}\n"
    score = correct_moves + correct_values
    max_score = 2 * len(answers)
    return score, details, max_score


def minimax_min_node_2_test(minimax_min_node, name=""):
    answers = [((3, 0), -6), ((5, 5), -8), ((1, 5), -12), ((5, 2), -2), ((3, 4), -4)]
    correct_moves = 0
    correct_values = 0
    details = ""
    for i, board in enumerate(BIG_BOARDS[:len(answers)]):
        try:
            move, value = minimax_min_node(board, 2, 1, 0)
        except Exception as e:
            details += f"Board {i}: Exception {e}\n"
            continue
        expected_move, expected_value = answers[i]
        if move == expected_move:
            correct_moves += 1
        else:
            details += f"Board {i}: expected move {expected_move}, got {move}\n"
        if value == expected_value:
            correct_values += 1
        else:
            details += f"Board {i}: expected value {expected_value}, got {value}\n"
    score = correct_moves + correct_values
    max_score = 2 * len(answers)
    return score, details, max_score


def minimax_max_node_2_test(minimax_max_node, name=""):
    # Testing selected boards (indices 1, 2, and 4)
    answers = {1: ((1, 1), 4), 2: ((3, 0), 6), 4: ((5, 2), 6)}
    correct_moves = 0
    correct_values = 0
    details = ""
    for i in sorted(answers.keys()):
        board = BIG_BOARDS[i]
        try:
            move, value = minimax_max_node(board, 2, 1, 0)
        except Exception as e:
            details += f"Board {i}: Exception {e}\n"
            continue
        expected_move, expected_value = answers[i]
        if move == expected_move:
            correct_moves += 1
        else:
            details += f"Board {i}: expected move {expected_move}, got {move}\n"
        if value == expected_value:
            correct_values += 1
        else:
            details += f"Board {i}: expected value {expected_value}, got {value}\n"
    score = correct_moves + correct_values
    max_score = 2 * len(answers)
    return score, details, max_score

