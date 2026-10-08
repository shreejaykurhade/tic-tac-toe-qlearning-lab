"""A small NumPy DQN for the same Tic-Tac-Toe states and rewards."""

from collections import deque
import random

import numpy as np

from .core import Board, encode_state, legal_actions


def state_vector(state: str) -> np.ndarray:
    """Encode nine squares as 18 inputs: AI marks, then opponent marks."""
    return np.array([float(cell == "1") for cell in state] +
                    [float(cell == "2") for cell in state], dtype=np.float32)


class DQNAgent:
    """Neural Q-values with replay memory and a slowly updated target network."""

    def __init__(self, alpha: float = 0.001, gamma: float = 0.97, seed: int = 42,
                 memory_size: int = 20_000, batch_size: int = 64,
                 target_interval: int = 250):
        self.alpha, self.gamma = alpha, gamma
        self.batch_size, self.target_interval = batch_size, target_interval
        self.rng = random.Random(seed)
        weights = np.random.default_rng(seed)
        sizes = (18, 64, 64, 9)
        # The online network learns; the target network provides stable targets.
        self.online = {
            f"{kind}{i}": (weights.normal(0, (2 / sizes[i]) ** 0.5,
                                          (sizes[i], sizes[i + 1])).astype(np.float32)
                           if kind == "W" else np.zeros(sizes[i + 1], np.float32))
            for i in range(3) for kind in ("W", "b")
        }
        self.target = {key: value.copy() for key, value in self.online.items()}
        self.first = {key: np.zeros_like(value) for key, value in self.online.items()}
        self.second = {key: np.zeros_like(value) for key, value in self.online.items()}
        self.memory = deque(maxlen=memory_size)
        self.updates = 0

    @staticmethod
    def _forward(inputs: np.ndarray, network: dict[str, np.ndarray]):
        hidden1 = np.maximum(inputs @ network["W0"] + network["b0"], 0)
        hidden2 = np.maximum(hidden1 @ network["W1"] + network["b1"], 0)
        values = hidden2 @ network["W2"] + network["b2"]
        return hidden1, hidden2, values

    def values(self, state: str) -> list[float]:
        """Predict one Q-value for each board square."""
        inputs = state_vector(state)[None, :]
        return self._forward(inputs, self.online)[-1][0].tolist()

    def select_action(self, board: Board, agent_mark: int, epsilon: float = 0.0,
                      rng: random.Random | None = None) -> int:
        """Explore during training; otherwise choose the best legal square."""
        rng = rng or self.rng
        actions = legal_actions(board)
        if not actions:
            raise ValueError("No legal action in a terminal state.")
        if rng.random() < epsilon:
            return rng.choice(actions)
        values = self.values(encode_state(board, agent_mark))
        best = max(values[action] for action in actions)
        return rng.choice([action for action in actions
                           if abs(values[action] - best) <= 1e-7])

    def remember(self, state: str, action: int, reward: float,
                 next_state: str | None, done: bool) -> None:
        """Store an AI move and the resulting board after the opponent replies."""
        if state[action] != "0":
            raise ValueError("Cannot learn an occupied action.")
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self) -> float | None:
        """Learn from a random replay batch; return its Huber loss."""
        if len(self.memory) < self.batch_size:
            return None
        batch = self.rng.sample(self.memory, self.batch_size)
        states = np.stack([state_vector(item[0]) for item in batch])
        actions = np.array([item[1] for item in batch], dtype=np.int64)
        rewards = np.array([item[2] for item in batch], dtype=np.float32)
        done = np.array([item[4] for item in batch], dtype=bool)
        next_states = [item[3] for item in batch]

        # Target Q uses a frozen network and ignores occupied squares.
        next_inputs = np.stack([state_vector(state or "000000000") for state in next_states])
        next_values = self._forward(next_inputs, self.target)[-1]
        legal = np.array([[cell == "0" for cell in (state or "111111111")]
                          for state in next_states])
        next_values = np.where(legal, next_values, -np.inf)
        next_best = np.max(next_values, axis=1)
        next_best[done] = 0.0
        targets = rewards + self.gamma * next_best

        hidden1, hidden2, predictions = self._forward(states, self.online)
        error = predictions[np.arange(self.batch_size), actions] - targets
        absolute = np.abs(error)
        loss = np.where(absolute <= 1, 0.5 * error ** 2, absolute - 0.5).mean()
        # Huber loss limits large errors; backpropagation changes all layers.
        output_gradient = np.zeros_like(predictions)
        output_gradient[np.arange(self.batch_size), actions] = np.clip(error, -1, 1) / self.batch_size
        middle_gradient = (output_gradient @ self.online["W2"].T) * (hidden2 > 0)
        first_gradient = (middle_gradient @ self.online["W1"].T) * (hidden1 > 0)
        gradients = {
            "W2": hidden2.T @ output_gradient, "b2": output_gradient.sum(axis=0),
            "W1": hidden1.T @ middle_gradient, "b1": middle_gradient.sum(axis=0),
            "W0": states.T @ first_gradient, "b0": first_gradient.sum(axis=0),
        }
        # Adam keeps a running average of gradients for steady learning.
        self.updates += 1
        for key, gradient in gradients.items():
            self.first[key] = 0.9 * self.first[key] + 0.1 * gradient
            self.second[key] = 0.999 * self.second[key] + 0.001 * gradient ** 2
            mean = self.first[key] / (1 - 0.9 ** self.updates)
            variance = self.second[key] / (1 - 0.999 ** self.updates)
            self.online[key] -= self.alpha * mean / (np.sqrt(variance) + 1e-8)
        if self.updates % self.target_interval == 0:
            self.target = {key: value.copy() for key, value in self.online.items()}
        return float(loss)
