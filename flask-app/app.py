from __future__ import annotations

import os
import sys
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
from flask import Flask, jsonify, render_template, request, session

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game2048.agent import DQNAgent
from game2048.game import Game2048

# Flask app configuration
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "2048-dqn-secret-key")

# Global variables
current_agent: Optional[DQNAgent] = None
current_game: Game2048 = Game2048()
game_history: List[Dict[str, Any]] = []
auto_play_active: bool = False

# Constants
CHANNELS = 16
ACTION_TO_DIRECTION = {0: "up", 1: "left", 2: "right", 3: "down"}
DIRECTION_TO_ACTION = {"up": 0, "left": 1, "right": 2, "down": 3}

def one_hot_state(board: np.ndarray, channels: int = CHANNELS) -> np.ndarray:
    """Convert board to one-hot encoded state."""
    state = np.zeros((1, 4, 4, channels), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            tile = int(board[i, j])
            if tile > 0:
                idx = int(np.log2(tile)) - 1
                if 0 <= idx < channels:
                    state[0, i, j, idx] = 1.0
    return state


def find_best_checkpoint(model_dir: str) -> Optional[str]:
    """Automatically find the best checkpoint in the given directory."""
    model_path = Path(model_dir)
    if not model_path.exists():
        return None
    
    # Look for checkpoint files
    patterns = [
        "dqn_checkpoint.keras",
        "dqn_latest.keras", 
        "dqn_ep*.keras",
        "*.keras"
    ]
    
    for pattern in patterns:
        matches = list(model_path.glob(pattern))
        if matches:
            # Sort by modification time, return the most recent
            best_match = max(matches, key=lambda p: p.stat().st_mtime)
            return str(best_match)
    
    # Look in subdirectories
    for subdir in model_path.iterdir():
        if subdir.is_dir():
            checkpoint = find_best_checkpoint(str(subdir))
            if checkpoint:
                return checkpoint
    
    return None


def load_agent_from_path(model_dir: str) -> tuple[bool, str]:
    """Load agent from model directory. Returns (success, message)."""
    global current_agent
    
    try:
        checkpoint_path = find_best_checkpoint(model_dir)
        if not checkpoint_path:
            return False, f"No valid checkpoint found in {model_dir}"
        
        # Create agent and load checkpoint
        agent = DQNAgent(load_weights=False, model_dir=str(Path(checkpoint_path).parent))
        agent.load(checkpoint_path)
        current_agent = agent
        
        return True, f"Successfully loaded: {Path(checkpoint_path).name}"
    
    except Exception as e:
        current_agent = None
        return False, f"Error loading model: {str(e)}"


def get_game_state() -> Dict[str, Any]:
    """Get current game state as JSON-serializable dict."""
    return {
        "board": current_game.board.astype(int).tolist(),
        "score": int(current_game.score),
        "max_tile": int(np.max(current_game.board)),
        "done": bool(current_game.done),
        "moves": len(game_history),
        "agent_loaded": current_agent is not None,
        "auto_play_active": auto_play_active,
        "agent_epsilon": float(current_agent.epsilon) if current_agent else None
    }


def add_to_history(move_type: str, direction: str, reward: float, changed: bool) -> None:
    """Add move to game history."""
    global game_history
    entry = {
        "move_type": move_type,
        "direction": direction,
        "score": int(current_game.score),
        "reward": float(reward),
        "max_tile": int(np.max(current_game.board)),
        "changed": bool(changed),
        "move_number": len(game_history) + 1
    }
    game_history.append(entry)
    
    # Keep only last 100 moves to prevent memory issues
    if len(game_history) > 100:
        game_history = game_history[-100:]


@app.route("/")
def index():
    """Main game page."""
    return render_template("index.html")


@app.route("/api/state")
def get_state():
    """Get current game state."""
    return jsonify(get_game_state())


@app.route("/api/reset", methods=["POST"])
def reset_game():
    """Reset the game to initial state."""
    global current_game, game_history, auto_play_active
    
    current_game = Game2048()
    game_history = []
    auto_play_active = False
    
    state = get_game_state()
    state["message"] = "Game reset successfully!"
    return jsonify(state)


@app.route("/api/move", methods=["POST"])
def manual_move():
    """Handle manual move from user."""
    global auto_play_active
    
    if current_game.done:
        return jsonify({
            "error": "Game is over! Please reset to continue.",
            **get_game_state()
        }), 400
    
    data = request.get_json() or {}
    direction = data.get("direction", "").lower()
    
    if direction not in DIRECTION_TO_ACTION:
        return jsonify({
            "error": "Invalid direction. Use: up, down, left, right",
            **get_game_state()
        }), 400
    
    # Stop auto play if active
    auto_play_active = False
    
    # Make the move
    changed, reward = current_game.move(direction)
    add_to_history("manual", direction, reward, changed)
    
    state = get_game_state()
    if not changed:
        state["warning"] = "Move had no effect!"
    if current_game.done:
        state["message"] = f"Game Over! Final score: {current_game.score}"
    
    return jsonify(state)


@app.route("/api/auto_move", methods=["POST"])
def auto_move():
    """Let AI make one move."""
    if not current_agent:
        return jsonify({
            "error": "No AI model loaded! Please load a model first.",
            **get_game_state()
        }), 400
    
    if current_game.done:
        return jsonify({
            "error": "Game is over! Please reset to continue.",
            **get_game_state()
        }), 400
    
    # Get AI action
    state_tensor = one_hot_state(current_game.board)
    action = current_agent.get_action(state_tensor, deterministic=True)
    direction = ACTION_TO_DIRECTION.get(action, "up")
    
    # Make the move
    changed, reward = current_game.move(direction)
    add_to_history("ai", direction, reward, changed)
    
    state = get_game_state()
    state["ai_move"] = {
        "direction": direction,
        "action": int(action),
        "reward": float(reward),
        "changed": bool(changed)
    }
    
    if not changed:
        state["warning"] = "AI made an invalid move!"
    if current_game.done:
        state["message"] = f"AI finished the game! Final score: {current_game.score}"
    
    return jsonify(state)


@app.route("/api/auto_play", methods=["POST"])
def toggle_auto_play():
    """Toggle automatic play mode."""
    global auto_play_active
    
    if not current_agent:
        return jsonify({
            "error": "No AI model loaded! Please load a model first.",
            **get_game_state()
        }), 400
    
    data = request.get_json() or {}
    auto_play_active = data.get("enable", not auto_play_active)
    
    state = get_game_state()
    state["message"] = f"Auto play {'enabled' if auto_play_active else 'disabled'}"
    return jsonify(state)


@app.route("/api/load_model", methods=["POST"])
def load_model():
    """Load AI model from specified directory."""
    data = request.get_json() or {}
    model_dir = data.get("model_dir", "").strip()
    
    if not model_dir:
        return jsonify({
            "error": "Please provide a model directory path",
            **get_game_state()
        }), 400
    
    # Expand relative paths
    if not os.path.isabs(model_dir):
        model_dir = str(PROJECT_ROOT / model_dir)
    
    success, message = load_agent_from_path(model_dir)
    
    state = get_game_state()
    if success:
        state["message"] = message
        return jsonify(state)
    else:
        state["error"] = message
        return jsonify(state), 400


@app.route("/api/history")
def get_history():
    """Get game move history."""
    return jsonify({
        "history": game_history,
        "total_moves": len(game_history)
    })


if __name__ == "__main__":
    # Try to load default model if available
    default_model_path = PROJECT_ROOT / "Model"
    if default_model_path.exists():
        success, msg = load_agent_from_path(str(default_model_path))
        if success:
            print(f"✅ {msg}")
        else:
            print(f"⚠️  {msg}")
    
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "1") == "1"
    
    print(f"🎮 Starting 2048 DQN Flask App on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)