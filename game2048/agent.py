from __future__ import annotations

import os, random
from collections import deque
from typing import Deque, Tuple, Optional

import numpy as np
import tensorflow as tf
import pandas as pd

# Constants
DEFAULT_INPUT_SHAPE = (4, 4, 16)  # (H, W, C)

# ✅ point to root/Model/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "Model")

class ReplayBuffer:
    """Simple replay buffer using numpy arrays stored in Python lists.
    Stores transitions as tuples: (state, action, reward, next_state, done)
    """
    def __init__(self, capacity: int = 100_000):
        self.capacity = int(capacity)
        self.buffer: Deque = deque(maxlen=self.capacity)
    #######################################
    def __len__(self) -> int:
        return len(self.buffer)
    #######################################
    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state.astype(np.float32), int(action), float(reward), next_state.astype(np.float32), bool(done)))
    #######################################
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.concatenate(states, axis=0),
                np.asarray(actions, dtype=np.int32),
                np.asarray(rewards, dtype=np.float32),
                np.concatenate(next_states, axis=0),
                np.asarray(dones, dtype=np.float32))

class DQNAgent:
    """Deep Q-Network agent for 2048.
    High-level API:
        - agent.act(state): returns action (0..3)
        - agent.remember(...): store transition
        - agent.train_step(batch_size): run a single training update from buffer
        - agent.save(path) / agent.load(path)
    Notes:
        - Input state expected shape: (batch, 4, 4, 16) and dtype float32.
        - Action mapping should match Game2048: e.g. 0=up,1=left,2=right,3=down (or adapt externally)
    """
    #######################################
    def __init__(self,
                input_shape: Tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
                lr: float = 1e-3,
                gamma: float = 0.99,
                epsilon_start: float = 1.0,
                epsilon_end: float = 0.01,
                epsilon_decay_steps: int = 200_000,
                replay_capacity: int = 100_000,
                batch_size: int = 64,
                target_update_freq: int = 1000,
                model_dir: Optional[str] = None,
                load_weights: bool = False):
        self.input_shape = input_shape
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_decay_steps = int(epsilon_decay_steps)
        self.epsilon_step = 0
        self.batch_size = int(batch_size)
        self.target_update_freq = int(target_update_freq)
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Replay buffer
        self.replay = ReplayBuffer(capacity=replay_capacity)
        
        # Optimizer and loss
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr)
        self.loss_fn = tf.keras.losses.Huber()
        
        # Main and target networks
        self.model = self._build_model()
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        
        # Bookkeeping
        self.train_steps = 0
        if load_weights:
            self.load(os.path.join(self.model_dir, "dqn.h5"))
    #######################################
    def _build_model(self) -> tf.keras.Model:
        """Build a small conv-net appropriate for the 4x4x16 input.
        The network is intentionally compact to keep inference/training stable.
        """
        inputs = tf.keras.layers.Input(shape=self.input_shape, dtype=tf.float32)
        # Two small parallel conv streams to capture horizontal and vertical patterns
        x1 = tf.keras.layers.Conv2D(128, kernel_size=(1, 2), activation='relu', padding='valid')(inputs)
        x1 = tf.keras.layers.Conv2D(128, kernel_size=(1, 2), activation='relu', padding='valid')(x1)
        
        x2 = tf.keras.layers.Conv2D(128, kernel_size=(2, 1), activation='relu', padding='valid')(inputs)
        x2 = tf.keras.layers.Conv2D(128, kernel_size=(2, 1), activation='relu', padding='valid')(x2)
        
        merged = tf.keras.layers.Concatenate()([tf.keras.layers.Flatten()(x1), tf.keras.layers.Flatten()(x2)])
        fc = tf.keras.layers.Dense(512, activation='relu')(merged)
        fc = tf.keras.layers.Dropout(0.25)(fc)
        fc = tf.keras.layers.Dense(256, activation='relu')(fc)
        outputs = tf.keras.layers.Dense(4, activation=None)(fc)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model
    #######################################
    def update_epsilon(self) -> None:
        """Linear epsilon decay per call."""
        if self.epsilon_step < self.epsilon_decay_steps:
            frac = self.epsilon_step / float(self.epsilon_decay_steps)
            self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)
            self.epsilon_step += 1
        else:
            self.epsilon = self.epsilon_end
    #######################################
    def act(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Return an action for the given state.
        Args:
            state: np.ndarray shape (1, 4, 4, 16)
            deterministic: if True use greedy action (no epsilon)
        """
        assert state.ndim == 4 and state.shape[1:] == self.input_shape, f"Expected state shape (1,{self.input_shape}), got {state.shape}"
        if (not deterministic) and (random.random() < self.epsilon):
            return random.randrange(4)
        q = self.model.predict(state, verbose=0)[0]
        return int(np.argmax(q))
    #######################################
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Alias for `act` to match common agent APIs while retaining epsilon-greedy logic."""
        return self.act(state, deterministic=deterministic)
    #######################################
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.replay.add(state, action, reward, next_state, done)
    #######################################
    def train_step(self, batch_size: Optional[int] = None) -> Optional[float]:
        """Sample a minibatch from replay and perform one gradient update.
        Returns the training loss (float) or None if not enough samples.
        """
        if batch_size is None:
            batch_size = self.batch_size
        if len(self.replay) < max(1, batch_size // 2):
            return None
        states, actions, rewards, next_states, dones = self.replay.sample(batch_size)
        # Predict Q(s') with target model for stability
        target_q_values = self.target_model.predict(next_states, verbose=0)
        max_next_q = np.max(target_q_values, axis=1)
        targets = rewards + (1.0 - dones) * (self.gamma * max_next_q)
        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)
            # gather the q-values for the taken actions
            action_masks = tf.one_hot(actions, 4, dtype=tf.float32)
            q_action = tf.reduce_sum(q_values * action_masks, axis=1)
            loss = self.loss_fn(targets, q_action)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        # Periodically update the target network
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())
        # decay epsilon
        self.update_epsilon()
        
        return float(loss.numpy())
    #######################################
    def save(self, path: Optional[str] = None) -> str:
        """Save model weights to path or default model_dir/dqn_checkpoint.keras"""
        path = path or os.path.join(self.model_dir, "dqn_checkpoint.keras")
        self.model.save(path)
        return path
    #######################################
    def load(self, path: str) -> None:
        """Load model from path. Will set both main and target networks."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        self.model = tf.keras.models.load_model(path)
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        # re-create optimizer and compile for consistency
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr)
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        print(f"Successfully loaded checkpoint from: {path}")
    #######################################
    # Optional: utility to load conv weights from CSVs when you have them
    def load_weights_from_csv(self, csv_map: dict) -> None:
        """Load weights provided as CSV files. csv_map: {layer_name: path_to_csv}
        This is a helper for non-standard weight formats. It attempts to reshape arrays
        to the expected layer weight shapes. Use with caution; prefer `load()` with
        Keras HDF5 models when possible.
        """
        for layer_name, csv_path in csv_map.items():
            if not os.path.exists(csv_path):
                raise FileNotFoundError(csv_path)
            df = pd.read_csv(csv_path)
            # Assume 'Weight' column exists and flatten ordering matches Keras.
            if 'Weight' not in df.columns:
                raise ValueError(f"CSV {csv_path} must contain a 'Weight' column")
            arr = df['Weight'].values.astype(np.float32)
            # find layer
            try:
                layer = self.model.get_layer(layer_name)
            except ValueError:
                raise ValueError(f"Layer {layer_name} not found in model")
            shapes = [w.shape for w in layer.get_weights()]
            # If shapes single weight (kernel) or [kernel,bias]
            if len(shapes) == 0:
                continue
            # Try to reshape arr into the kernel shape
            kernel_shape = shapes[0]
            expected_size = int(np.prod(kernel_shape))
            if arr.size != expected_size and arr.size != expected_size + shapes[1][0]:
                raise ValueError(f"CSV size {arr.size} doesn't match expected {expected_size} for layer {layer_name}")
            kernel = arr[:expected_size].reshape(kernel_shape)
            if len(shapes) == 1:
                layer.set_weights([kernel])
            else:
                bias = arr[expected_size:expected_size + shapes[1][0]] if arr.size >= expected_size + shapes[1][0] else np.zeros(shapes[1], dtype=np.float32)
                layer.set_weights([kernel, bias])
    #######################################
    # Convenience: run a training loop over episodes
    def fit(self,
            env,
            episodes: int = 10_000,
            max_steps_per_episode: int = 1_000,
            start_train_after: int = 1_000,
            train_every: int = 1,
            save_every: int = 1_000,
            verbose: int = 1):
        """High-level training loop. `env` must implement:
            - reset() -> state (1,4,4,16)
            - step(action) -> next_state, reward, done, info
        """
        best_score = -float('inf')
        for ep in range(1, episodes + 1):
            state = env.reset()
            # If env.reset returns board, convert to state using env.get_state() if available
            if isinstance(state, np.ndarray) and state.shape != (1, *self.input_shape):
                # try env.get_state()
                if hasattr(env, 'get_state'):
                    state = env.get_state()
                else:
                    # expand dims
                    state = np.expand_dims(state.astype(np.float32), axis=0)
            total_reward = 0.0
            for t in range(max_steps_per_episode):
                action = self.act(state)
                next_board, reward, done, info = env.step(action)
                # If env.step returns board array, convert to state
                if isinstance(next_board, np.ndarray) and next_board.shape != (1, *self.input_shape):
                    if hasattr(env, 'get_state'):
                        next_state = env.get_state()
                    else:
                        next_state = np.expand_dims(next_board.astype(np.float32), axis=0)
                else:
                    next_state = next_board
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += float(reward)
                # Training
                if (len(self.replay) >= start_train_after) and ((t % train_every) == 0):
                    loss = self.train_step()
                if done:
                    break
            # Optionally save
            if ep % save_every == 0:
                self.save(os.path.join(self.model_dir, f"dqn_ep{ep}.h5"))
            if verbose and (ep % max(1, episodes // 20) == 0):
                print(f"Episode {ep:6d} | Reward {total_reward:.2f} | Epsilon {self.epsilon:.4f} | Buffer {len(self.replay)}")
        return