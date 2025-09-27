# run: python tests/test_agent.py
import os, sys
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"   # suppress INFO and WARNING logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # disable oneDNN custom ops warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game2048.agent import DQNAgent
from game2048.game import Game2048

SEPARATOR = "-" * 20

def print_section(title: str) -> None:
    print(title)
    print(SEPARATOR)


def display_board(board: np.ndarray, title: str) -> None:
    print_section(title)
    print(board)
    print()

def preprocess_state(board: np.ndarray, channels: int = 16) -> np.ndarray:
    """Convert a 4x4 board of tile values into one-hot planes of shape (1,4,4,16).
    Plane k corresponds to tile 2^(k+1):
    - tile 2  -> plane 0
    - tile 4  -> plane 1
    - tile 8  -> plane 2
    - ... up to plane 15
    Empty cells (0) are represented by all-zero channels.
    """
    assert board.shape == (4, 4), f"Expected board shape (4,4), got {board.shape}"
    state = np.zeros((1, 4, 4, channels), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            v = int(board[i, j])
            if v > 0:
                idx = int(np.log2(v)) - 1  # 2->0, 4->1, 8->2, ...
                if 0 <= idx < channels:
                    state[0, i, j, idx] = 1.0
                # If idx is out of range (very large tile), ignore/clamp implicitly
    return state

def test_agent_action():
    """Test if the agent can choose an action for a given state."""
    print('- '*10)
    print("TEST AGENT ACTION")
    print('- '*10)
    env = Game2048()
    env.reset()
    agent = DQNAgent()
    
    # Prepare state
    if hasattr(env, "get_state"):
        state = env.get_state()
        # Fallback if get_state doesn't return the expected one-hot shape
        if not (isinstance(state, np.ndarray) and state.shape == (1, 4, 4, 16)):
            state = preprocess_state(env.board)
    else:
        state = preprocess_state(env.board)
    
    # Agent chooses action
    action = agent.get_action(state)
    display_board(env.board, "Current game board")
    print(f"Chosen action index: {action}")

def test_agent_training_step():
    """Test if the agent can perform a single training step."""
    print()
    print('- '*10)
    print("TEST AGENT TRAINING STEP")
    print('- '*10)
    env = Game2048()
    agent = DQNAgent()
    state = env.reset()
    if hasattr(env, "get_state"):
        state = env.get_state()
        if not (isinstance(state, np.ndarray) and state.shape == (1, 4, 4, 16)):
            state = preprocess_state(env.board)
    else:
        state = preprocess_state(state)
    
    # One step interaction
    display_board(env.board, "Board before step")
    action = agent.get_action(state)
    next_board, reward, done, info = env.step(action)
    display_board(next_board, "Board after step")
    print(f"Action: {action} | Reward: {reward} | Done: {done} | Info: {info}")
    
    if hasattr(env, "get_state"):
        next_state = env.get_state()
        if not (isinstance(next_state, np.ndarray) and next_state.shape == (1, 4, 4, 16)):
            next_state = preprocess_state(next_board)
    else:
        next_state = preprocess_state(next_board)
    
    # Store transition in replay buffer
    agent.remember(state, action, reward, next_state, done)
    
    # Run one training step
    loss = agent.train_step(batch_size=1)
    print(f"Training loss: {loss}")


if __name__ == "__main__":
    print('-'*40)
    print("RUNNING AGENT CHECKS")
    print('-'*40)
    test_agent_action()
    test_agent_training_step()
    print('-'*40)
    print("AGENT CHECKS COMPLETED")
    print('-'*40)