# run: python game2048/train.py --episodes=1000 --log-every=10 --save-every=100 --eval-every=200
from __future__ import annotations

import os, sys, time, math, json, argparse, datetime
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Tuple

# os.add_dll_directory(r"C:\Users\abdal\anaconda3\envs\2048-rl-env\Library\bin")
# TensorFlow settings
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

# GPU memory growth (to avoid OOM)
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game2048.agent import DQNAgent
from game2048.game import Game2048

# ============================================================
# Config
# ============================================================
@dataclass
class TrainingConfig:
    episodes: int = 1_000
    max_steps_per_episode: int = 1_000
    batch_size: int = 128
    start_train_after: int = 1_000
    train_every: int = 4
    log_every: int = 10
    save_every: int = 100
    eval_every: int = 200
    model_dir: str = os.path.join(PROJECT_ROOT, "Model")
    checkpoint_name: str = "dqn_checkpoint.keras"
    invalid_move_penalty: float = 1.0
    target_tile_bonus: float = 50.0
    target_tile: int = 2048
    evaluation_episodes: int = 3
    early_stop_patience: int = 5  # stop if no eval improvement

@dataclass
class TrainingSummary:
    best_score: float
    duration_seconds: float


# ============================================================
# Helpers
# ============================================================
def preprocess_board(board: np.ndarray, channels: int = 16) -> np.ndarray:
    """Convert a board of tile values into one-hot encoded planes."""
    if board.shape != (4, 4):
        raise ValueError(f"Expected board shape (4, 4), received {board.shape}")
    state = np.zeros((1, 4, 4, channels), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            tile = int(board[i, j])
            if tile > 0:
                idx = int(math.log(tile, 2)) - 1
                if 0 <= idx < channels:
                    state[0, i, j, idx] = 1.0
    return state

def get_env_state(env: Game2048, channels: int = 16) -> np.ndarray:
    if hasattr(env, "get_state"):
        state = env.get_state(channels=channels)
        if isinstance(state, np.ndarray) and state.shape == (1, 4, 4, channels):
            return state
    return preprocess_board(env.board, channels=channels)

def compute_reward(raw_reward: float, info: Dict[str, object], done: bool, config: TrainingConfig) -> float:
    reward = float(raw_reward)
    if not info.get("changed", True):
        reward -= config.invalid_move_penalty
    if done:
        max_tile = info.get("max_tile")
        if max_tile is None:
            max_tile = int(np.max(info.get("board", 0))) if isinstance(info.get("board"), np.ndarray) else None
        if max_tile and max_tile >= config.target_tile:
            reward += config.target_tile_bonus
    return reward

def maybe_train(agent: DQNAgent, step_count: int, config: TrainingConfig) -> Optional[float]:
    if len(agent.replay) >= config.start_train_after and (step_count % config.train_every == 0):
        return agent.train_step(batch_size=config.batch_size)
    agent.update_epsilon()
    return None

# ============================================================
# Core training
# ============================================================
def run_episode(env: Game2048, agent: DQNAgent, config: TrainingConfig) -> Dict[str, float]:
    env.reset()
    state = get_env_state(env)
    total_reward, steps = 0.0, 0
    losses: List[float] = []
    for step in range(1, config.max_steps_per_episode + 1):
        action = agent.get_action(state)
        next_board, raw_reward, done, info = env.step(action)
        info.update({"max_tile": int(np.max(env.board)), "board": env.board.copy()})
        shaped_reward = compute_reward(raw_reward, info, done, config)
        next_state = get_env_state(env)
        agent.remember(state, action, shaped_reward, next_state, done)
        loss = maybe_train(agent, step, config)
        if loss is not None:
            losses.append(loss)
        state = next_state
        total_reward += shaped_reward
        steps = step
        if done:
            break
    return {
        "score": float(env.score),
        "reward": total_reward,
        "loss": float(np.mean(losses)) if losses else math.nan,
        "steps": float(steps),
        "epsilon": float(agent.epsilon),
        "max_tile": float(np.max(env.board)),
    }

def evaluate_agent(agent: DQNAgent, episodes: int, max_steps: int) -> Dict[str, float]:
    env = Game2048()
    scores, max_tiles = [], []
    for _ in range(episodes):
        env.reset()
        state = get_env_state(env)
        score = 0.0
        for _ in range(max_steps):
            action = agent.get_action(state, deterministic=True)
            _, reward, done, _ = env.step(action)
            state = get_env_state(env)
            score += reward
            if done:
                break
        scores.append(score)
        max_tiles.append(np.max(env.board))
    return {
        "eval_score": float(np.mean(scores)),
        "eval_max_tile": float(np.mean(max_tiles)),
    }

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def log_episode(episode: int, metrics: Dict[str, float]) -> None:
    safe_loss = "{:.4f}".format(metrics["loss"]) if not math.isnan(metrics["loss"]) else "n/a"
    print(
        f"{'Episode':<10}{'Score':<12}{'Reward':<12}{'Loss':<12}{'Steps':<10}{'Epsilon':<12}{'Max tile':<10}")
    print(
        f"{episode:<10}{metrics['score']:<12.1f}{metrics['reward']:<12.1f}"
        f"{safe_loss:<12}{metrics['steps']:<10}{metrics['epsilon']:<12.4f}{metrics['max_tile']:<10}")

def save_metrics(path: Path, history: List[Dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

# ============================================================
# Training loop
# ============================================================
def run_training_session(
    config: TrainingConfig,
    *,
    agent: Optional[DQNAgent] = None,
    persist_artifacts: bool = True,
    capture_history: bool = True,
    progress_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> Tuple[DQNAgent, List[Dict[str, float]], TrainingSummary]:
    # Setup experiment folder
    if agent is None:
        # Create new experiment folder for fresh training
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(config.model_dir, f"run-{timestamp}")
        ensure_dir(run_dir)
        checkpoint_path = os.path.join(run_dir, config.checkpoint_name)
        agent = DQNAgent(batch_size=config.batch_size, model_dir=run_dir)
        print("Starting fresh training session...")
    else:
        # Use existing agent's model directory for resumed training
        run_dir = agent.model_dir
        checkpoint_path = os.path.join(run_dir, config.checkpoint_name)
        print(f"Resuming training from checkpoint: {checkpoint_path}")
    
    env = Game2048()
    history: List[Dict[str, float]] = []
    
    # Load existing history if resuming
    history_file = Path(run_dir) / "training_history.json"
    if agent is not None and history_file.exists():
        try:
            import json
            with history_file.open("r") as f:
                existing_history = json.load(f)
            history.extend(existing_history)
            print(f"Loaded {len(existing_history)} previous episodes from history")
        except Exception as e:
            print(f"Could not load training history: {e}")
    best_score, best_eval, no_improve = -float("inf"), -float("inf"), 0
    # TensorBoard writer
    tb_writer = tf.summary.create_file_writer(os.path.join(run_dir, "logs"))
    start_time = time.time()
    start_episode = len(history) + 1 if history else 1
    
    for episode in range(start_episode, start_episode + config.episodes):
        metrics = run_episode(env, agent, config)
        entry = {"episode": episode, **metrics}
        if capture_history:
            history.append(entry)
        # log metrics
        if progress_callback is not None:
            progress_callback(episode, entry)
        if episode % config.log_every == 0 or episode == 1:
            log_episode(episode, metrics)
        with tb_writer.as_default():
            for k, v in metrics.items():
                tf.summary.scalar(k, v, step=episode)
        # save model
        if persist_artifacts and episode % config.save_every == 0:
            agent.save(os.path.join(run_dir, f"dqn_ep{episode}.keras"))
            if capture_history:
                save_metrics(Path(run_dir) / "training_history.json", history)
        # track best score
        if metrics["score"] > best_score:
            best_score = metrics["score"]
            if persist_artifacts:
                agent.save(checkpoint_path)
        # evaluation
        if config.eval_every and episode % config.eval_every == 0:
            eval_stats = evaluate_agent(agent, config.evaluation_episodes, config.max_steps_per_episode)
            print(
                f"--> Evaluation | Avg score: {eval_stats['eval_score']:.1f} "
                f"| Avg max tile: {eval_stats['eval_max_tile']:.0f}"
            )
            with tb_writer.as_default():
                for k, v in eval_stats.items():
                    tf.summary.scalar(k, v, step=episode)
            if eval_stats["eval_score"] > best_eval:
                best_eval = eval_stats["eval_score"]
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= config.early_stop_patience:
                print(f"Early stopping triggered at episode {episode}")
                break
    duration = time.time() - start_time
    summary = TrainingSummary(best_score=best_score, duration_seconds=duration)
    if persist_artifacts:
        agent.save(checkpoint_path)
        if capture_history:
            save_metrics(Path(run_dir) / "training_history.json", history)
    return agent, history, summary


# ============================================================
# CLI
# ============================================================
def train(config: TrainingConfig) -> TrainingSummary:
    _, _, summary = run_training_session(config)
    print(f"Training finished in {summary.duration_seconds/60:.2f} minutes. Best score: {summary.best_score:.1f}")
    return summary

def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train a DQN agent to play 2048.")
    parser.add_argument("--episodes", type=int, default=TrainingConfig.episodes)
    parser.add_argument("--max-steps", type=int, default=TrainingConfig.max_steps_per_episode)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--start-train-after", type=int, default=TrainingConfig.start_train_after)
    parser.add_argument("--train-every", type=int, default=TrainingConfig.train_every)
    parser.add_argument("--log-every", type=int, default=TrainingConfig.log_every)
    parser.add_argument("--save-every", type=int, default=TrainingConfig.save_every)
    parser.add_argument("--eval-every", type=int, default=TrainingConfig.eval_every)
    parser.add_argument("--model-dir", type=str, default=TrainingConfig.model_dir)
    parser.add_argument("--checkpoint-name", type=str, default=TrainingConfig.checkpoint_name)
    parser.add_argument("--invalid-move-penalty", type=float, default=TrainingConfig.invalid_move_penalty)
    parser.add_argument("--target-tile", type=int, default=TrainingConfig.target_tile)
    parser.add_argument("--target-tile-bonus", type=float, default=TrainingConfig.target_tile_bonus)
    parser.add_argument("--evaluation-episodes", type=int, default=TrainingConfig.evaluation_episodes)
    parser.add_argument("--early-stop-patience", type=int, default=TrainingConfig.early_stop_patience)
    args = parser.parse_args()
    return TrainingConfig(
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        batch_size=args.batch_size,
        start_train_after=args.start_train_after,
        train_every=args.train_every,
        log_every=args.log_every,
        save_every=args.save_every,
        eval_every=args.eval_every,
        model_dir=args.model_dir,
        checkpoint_name=args.checkpoint_name,
        invalid_move_penalty=args.invalid_move_penalty,
        target_tile=args.target_tile,
        target_tile_bonus=args.target_tile_bonus,
        evaluation_episodes=args.evaluation_episodes,
        early_stop_patience=args.early_stop_patience,
    )

def main() -> None:
    config = parse_args()
    print("Training configuration:")
    print(json.dumps(asdict(config), indent=2))
    train(config)

if __name__ == "__main__":
    main()