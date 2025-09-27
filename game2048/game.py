import random
import numpy as np

class Game2048:
    """Minimal and clean implementation of the 2048 game logic."""
    #######################################
    def __init__(self, size=4):
        self.size = size
        self.reset()
    #######################################
    def reset(self):
        """Reset the game state."""
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.done = False
        self.add_random_tile()
        self.add_random_tile()
        return self.board
    #######################################
    def add_random_tile(self):
        """Add a random tile (2 or 4) in an empty cell."""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if not empty_cells:
            return
        i, j = random.choice(empty_cells)
        self.board[i, j] = 4 if random.random() < 0.1 else 2
    #######################################
    def move(self, direction):
        """
        Make a move in one of the 4 directions.
        direction ∈ ["up", "down", "left", "right"].
        Returns (changed, score_gain).
        """
        if self.done:
            return False, 0
        
        rotated = False
        flipped = False
        board = self.board.copy()
        
        if direction == "up":
            board = np.transpose(board)
            rotated = True
        elif direction == "down":
            board = np.flip(np.transpose(board), axis=1)
            rotated = True
            flipped = True
        elif direction == "right":
            board = np.flip(board, axis=1)
            flipped = True
        elif direction != "left":
            raise ValueError("Invalid direction. Use up/down/left/right.")
        changed, score_gain, new_board = self._move_left(board)
        
        # Reverse transformations
        if rotated and flipped:
            new_board = np.transpose(np.flip(new_board, axis=1))
        elif rotated:
            new_board = np.transpose(new_board)
        elif flipped:
            new_board = np.flip(new_board, axis=1)
        
        if changed:
            self.board = new_board
            self.score += score_gain
            self.add_random_tile()
            self.done = self.is_done()
        return changed, score_gain
    #######################################
    def step(self, action):
        """Apply an action and return (board, reward, done, info)."""
        direction_map = {0: "up", 1: "left", 2: "right", 3: "down"}
        
        if isinstance(action, str):
            direction = action.lower()
        elif isinstance(action, (int, np.integer)):
            if action not in direction_map:
                raise ValueError(f"Invalid action index {action}; expected 0..3")
            direction = direction_map[int(action)]
        else:
            raise TypeError("Action must be an int (0..3) or direction string")
        
        changed, reward = self.move(direction)
        if not changed:
            reward = 0.0
        
        info = {"changed": changed, "direction": direction}
        return self.board.copy(), float(reward), bool(self.done), info
    #######################################
    def _move_left(self, board):
        """Helper: slide + merge to the left."""
        new_board = np.zeros_like(board)
        score_gain = 0
        changed = False
        for i in range(self.size):
            row = board[i][board[i] != 0]  # remove zeros
            new_row = []
            skip = False
            j = 0
            while j < len(row):
                if j + 1 < len(row) and row[j] == row[j + 1]:
                    new_val = row[j] * 2
                    score_gain += new_val
                    new_row.append(new_val)
                    skip = True
                    j += 2
                else:
                    new_row.append(row[j])
                    j += 1
            while len(new_row) < self.size:
                new_row.append(0)
            new_board[i] = new_row
            if not np.array_equal(new_board[i], board[i]):
                changed = True
        return changed, score_gain, new_board
    #######################################
    def is_done(self):
        """Check if no moves are possible."""
        if np.any(self.board == 0):
            return False
        # Check merges possible
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j + 1]:
                    return False
                if self.board[j, i] == self.board[j + 1, i]:
                    return False
        return True
    #######################################
    def get_state(self, channels: int = 16) -> np.ndarray:
        """Return a one-hot encoding of the board with shape (1, size, size, channels)."""
        if channels <= 0:
            raise ValueError("channels must be a positive integer")
        state = np.zeros((1, self.size, self.size, channels), dtype=np.float32)
        for i in range(self.size):
            for j in range(self.size):
                tile = int(self.board[i, j])
                if tile > 0:
                    idx = int(np.log2(tile)) - 1
                    if 0 <= idx < channels:
                        state[0, i, j, idx] = 1.0
        return state
    #######################################
    def __str__(self):
        return f"Score: {self.score}\n{self.board}"