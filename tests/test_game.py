# run: python tests/test_game.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from game2048.game import Game2048

SEPARATOR = "-" * 20

def print_section(title: str) -> None:
    print(title)
    print(SEPARATOR)


def display_board(board: np.ndarray, title: str) -> None:
    print_section(title)
    print(board)
    print()


def expect_value_error(func, description: str) -> None:
    try:
        func()
    except ValueError:
        print(f"✔️  ValueError raised as expected: {description}")
    else:
        raise AssertionError(f"Expected ValueError was not raised: {description}")


def test_simple_moves():
    print('-'*40)
    print("             TEST SIMPLE MOVES")
    print('-'*40)
    g = Game2048()
    g.add_random_tile = lambda: None  # avoid randomness during assertions
    starting_board = np.array([
        [2, 2, 0, 0],
        [4, 4, 2, 0],
        [2, 0, 2, 2],
        [0, 0, 0, 0],
    ])
    
    g.board = starting_board.copy()
    display_board(g.board, "1-Initial board")
    
    changed_left, score_left = g.move("left")
    expected_left = np.array([
        [4, 0, 0, 0],
        [8, 2, 0, 0],
        [4, 2, 0, 0],
        [0, 0, 0, 0],
    ])
    
    display_board(g.board, "- After moving LEFT")
    print(f"Changed: {changed_left}, Score gained: {score_left}")
    assert changed_left is True
    assert score_left == 16
    assert np.array_equal(g.board, expected_left)
    print('- -'*10)
    
    g.board = starting_board.copy()
    changed_right, score_right = g.move("right")
    expected_right = np.array([
        [0, 0, 0, 4],
        [0, 0, 8, 2],
        [0, 0, 2, 4],
        [0, 0, 0, 0],
    ])
    
    display_board(g.board, "- After moving RIGHT")
    print(f"Changed: {changed_right}, Score gained: {score_right}")
    
    assert changed_right is True
    assert score_right == 16
    assert np.array_equal(g.board, expected_right)
    print('- -'*10)
    print("✔️  Simple move logic confirmed")
    print('-'*40)


def test_game_over():
    print("            TEST GAME OVER")
    print('-'*40)
    g = Game2048()
    g.board = np.array([
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [4, 2, 4, 2],
    ])
    
    display_board(g.board, "- Filled board")
    assert g.is_done() is True
    print('- -'*10)
    print("✔️  No moves available, game is done")
    print('-'*40)

def test_get_state_encoding():
    print("            TEST STATE ENCODING")
    print('-'*40)
    g = Game2048()
    g.board = np.array([
        [0, 2, 4, 8],
        [16, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    
    display_board(g.board, "Board for encoding")
    state = g.get_state()
    
    print(f"State shape: {state.shape}")
    assert state.shape == (1, 4, 4, 16)
    assert state[0, 0, 1, 0] == 1.0  # tile 2 -> plane 0
    assert state[0, 0, 2, 1] == 1.0  # tile 4 -> plane 1
    assert state[0, 0, 3, 2] == 1.0  # tile 8 -> plane 2
    assert state[0, 1, 0, 3] == 1.0  # tile 16 -> plane 3
    assert np.isclose(state.sum(), 4.0)
    print('- -'*10)
    print("✔️  State encoding matches expected one-hot planes")
    expect_value_error(lambda: g.get_state(0), "channels must be positive")
    print('-'*40)


def run_all() -> None:
    test_simple_moves()
    test_game_over()
    test_get_state_encoding()
    print("ALL TESTS COMPLETED ✅")
    print('-'*40)


if __name__ == "__main__":
    run_all()