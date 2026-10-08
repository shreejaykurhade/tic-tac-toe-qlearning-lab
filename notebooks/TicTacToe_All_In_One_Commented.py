# One Python cell: board rules, tabular Q-learning, DQN, plots, and game.
# Part 1: board rules and training opponents.
"""Board rules and opponents used to train the Q-learning agent."""

from functools import lru_cache
import random
from typing import TypeAlias

Board: TypeAlias = tuple[int, ...]
# The tuple has nine squares in reading order: 0..2, 3..5, 6..8.
X, O = 1, 2
EMPTY_BOARD: Board = (0,) * 9
WIN_LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))


def legal_actions(board: Board) -> list[int]:
    """Return empty square numbers, or none if someone has won."""
    return [] if winner(board) else [i for i, mark in enumerate(board) if mark == 0]


@lru_cache(maxsize=None)
def winner(board: Board) -> int:
    """Check all eight possible winning lines; return X, O, or 0."""
    for a, b, c in WIN_LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return 0


def terminal(board: Board) -> bool:
    """Stop after a win or a full board."""
    return bool(winner(board)) or 0 not in board


def play(board: Board, action: int, mark: int) -> Board:
    """Make one move and return a new board, leaving the old board unchanged."""
    if len(board) != 9 or mark not in (X, O):
        raise ValueError("Expected a nine-cell board and mark X=1 or O=2.")
    if terminal(board):
        raise ValueError("Cannot play after a terminal state.")
    if action not in range(9) or board[action] != 0:
        raise ValueError("Action must select an empty cell in 0..8.")
    return board[:action] + (mark,) + board[action + 1:]


def encode_state(board: Board, agent_mark: int) -> str:
    """Describe the board from the agent's view for the Q-table key."""
    # Example: human X at top-left, AI O to move -> "200000000".
    # Using 1=self and 2=opponent lets the same Q-table serve X and O.
    return "".join("0" if m == 0 else "1" if m == agent_mark else "2" for m in board)


def other(mark: int) -> int:
    """Switch between X=1 and O=2."""
    return 3 - mark


def _symmetries() -> tuple[tuple[int, ...], ...]:
    # A rotated or mirrored board has the same strategy in new coordinates.
    # These eight mappings let us share one Q-table row across those boards.
    result = []
    for reflection in (False, True):
        for rotations in range(4):
            permutation = []
            for i in range(9):
                r, c = divmod(i, 3)
                if reflection:
                    c = 2 - c
                for _ in range(rotations):
                    r, c = c, 2 - r
                permutation.append(3 * r + c)
            result.append(tuple(permutation))
    return tuple(result)


SYMMETRIES = _symmetries()


def transform_state(state: str, permutation: tuple[int, ...]) -> str:
    """Move every cell to its rotated/reflected position."""
    result = ["0"] * 9
    for i, j in enumerate(permutation):
        result[j] = state[i]
    return "".join(result)


@lru_cache(maxsize=None)
def canonicalize(state: str) -> tuple[str, tuple[int, ...]]:
    """Pick one shared Q-table key for equivalent board orientations."""
    # Also return the mapping so action values can be shown on the real board.
    return min(((transform_state(state, p), p) for p in SYMMETRIES), key=lambda pair: pair[0])


@lru_cache(maxsize=None)
def minimax_value(board: Board, to_move: int) -> int:
    """Score a position assuming both sides make perfect moves."""
    # Minimax is an opponent for training/testing, not the playable AI policy.
    won = winner(board)
    if won:
        return 1 if won == to_move else -1
    actions = legal_actions(board)
    if not actions:
        return 0
    return max(-minimax_value(play(board, a, to_move), other(to_move)) for a in actions)


@lru_cache(maxsize=None)
def minimax_actions(board: Board, mark: int) -> tuple[int, ...]:
    """Find optimal replies for the training opponent."""
    actions = legal_actions(board)
    if not actions:
        return ()
    values = [-minimax_value(play(board, a, mark), other(mark)) for a in actions]
    best = max(values)
    return tuple(a for a, v in zip(actions, values) if v == best)


def random_action(board: Board, mark: int, rng: random.Random) -> int:
    return rng.choice(legal_actions(board))


def minimax_action(board: Board, mark: int, rng: random.Random) -> int:
    return rng.choice(minimax_actions(board, mark))


def tactical_action(board: Board, mark: int, rng: random.Random) -> int:
    """Training opponent: win now, block a win, then prefer strong squares."""
    actions = legal_actions(board)
    for target in (mark, other(mark)):
        winning = [a for a in actions if winner(play(board, a, target)) == target]
        if winning:
            return rng.choice(winning)
    if 4 in actions:
        return 4
    corners = [a for a in actions if a in (0, 2, 6, 8)]
    return rng.choice(corners or actions)


OPPONENTS = {"random": random_action, "tactical": tactical_action, "minimax": minimax_action}


def mixture_action(board: Board, mark: int, rng: random.Random,
                   weights: tuple[float, float, float]) -> int:
    """Sample random, tactical, or minimax for one opponent turn."""
    name = rng.choices(tuple(OPPONENTS), weights=weights, k=1)[0]
    return OPPONENTS[name](board, mark, rng)

# Part 2: Q-table, action selection, and learning update.
"""Learn one expected reward for each board state and legal move."""

import json
from pathlib import Path
import random



class QLearningAgent:
    def __init__(self, alpha: float = 0.15, gamma: float = 0.97, seed: int = 42):
        # alpha = update size; gamma = importance of future rewards.
        # Each state stores nine values, one for each board square.
        self.alpha = alpha
        self.gamma = gamma
        self.rng = random.Random(seed)
        self.q: dict[str, list[float]] = {}

    def values(self, state: str) -> list[float]:
        # Look up the shared rotated/mirrored board. An unseen board has zeros.
        # Map the stored values back to the squares the player can see.
        key, permutation = canonicalize(state)
        values = self.q.get(key, [0.0] * 9)
        return [values[permutation[a]] for a in range(9)]

    def greedy_actions(self, board: Board, mark: int) -> list[int]:
        # Find the highest Q-value among empty squares only.
        actions = legal_actions(board)
        if not actions:
            return []
        values = self.values(encode_state(board, mark))
        maximum = max(values[a] for a in actions)
        # Keep every equally good move; selection can break the tie randomly.
        return [a for a in actions if abs(values[a] - maximum) <= 1e-12]

    def select_action(self, board: Board, agent_mark: int, epsilon: float = 0.0,
                      rng: random.Random | None = None) -> int:
        rng = rng or self.rng
        actions = legal_actions(board)
        if not actions:
            raise ValueError("No legal action in a terminal state.")
        # Epsilon-greedy: try a random legal move with probability epsilon.
        # Otherwise choose a move with the highest learned value.
        if rng.random() < epsilon:
            return rng.choice(actions)
        return rng.choice(self.greedy_actions(board, agent_mark))

    def update(self, state: str, action: int, reward: float,
               next_state: str | None, done: bool) -> float:
        """Learn from one AI move and the opponent's reply."""
        if state[action] != "0":
            raise ValueError("Cannot learn an illegal action.")
        key, permutation = canonicalize(state)
        values = self.q.setdefault(key, [0.0] * 9)
        canonical_action = permutation[action]
        # Finished game: target = win/draw/loss reward.
        # Ongoing game: target = reward + gamma * best value next turn.
        target = reward
        if not done:
            if next_state is None:
                raise ValueError("A nonterminal update needs a next state.")
            next_values = self.values(next_state)
            actions = [a for a, cell in enumerate(next_state) if cell == "0"]
            if not actions:
                raise ValueError("A nonterminal next state needs a legal action.")
            target += self.gamma * max(next_values[a] for a in actions)
        td_error = target - values[canonical_action]
        # Move the old estimate partway toward the target:
        # new Q = old Q + alpha * (target - old Q).
        values[canonical_action] += self.alpha * td_error
        return abs(td_error)


# Part 3: DQN neural network, replay memory, and target network.
"""A small NumPy DQN for the same Tic-Tac-Toe states and rewards."""

from collections import deque
import random

import numpy as np



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

# Part 4: DQN training and comparison helpers.
"""Train and compare the neural DQN with the tabular Q-learning agent."""

from dataclasses import dataclass
import json
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np



@dataclass
class DQNConfig:
    episodes: int = 20_000
    seed: int = 42
    evaluation_games_per_role: int = 200
    replay_every: int = 4
    history_interval: int = 1_000


def train(config: DQNConfig, progress: bool = True):
    """Train a DQN using the same game and terminal rewards as tabular Q-learning."""
    agent = DQNAgent(seed=config.seed)
    opponent_rng = random.Random(config.seed + 1)
    history = []
    moves_seen = 0
    window = {"games": 0, "reward": 0.0, "non_loss": 0, "moves": 0,
              "loss": 0.0, "updates": 0}
    for episode in range(1, config.episodes + 1):
        mark = X if episode % 2 else O
        board, board_moves = EMPTY_BOARD, 0
        if mark == O:
            board = play(board, mixture_action(board, X, opponent_rng,
                                               (0.25, 0.15, 0.60)), X)
            board_moves += 1
        fraction = min(1.0, (episode - 1) / max(1, config.episodes * 0.85))
        epsilon = 1.0 + fraction * (0.03 - 1.0)
        while True:
            state = encode_state(board, mark)
            action = agent.select_action(board, mark, epsilon)
            board = play(board, action, mark)
            board_moves += 1
            if not terminal(board):
                reply = mixture_action(board, other(mark), opponent_rng,
                                       (0.25, 0.15, 0.60))
                board = play(board, reply, other(mark))
                board_moves += 1
            done = terminal(board)
            result = winner(board)
            reward = ((1.0 if result == mark else -1.0) if result else 0.3) if done else 0.0
            agent.remember(state, action, reward,
                           None if done else encode_state(board, mark), done)
            moves_seen += 1
            if moves_seen % config.replay_every == 0:
                loss = agent.train_step()
                if loss is not None:
                    window["loss"] += loss
                    window["updates"] += 1
            if done:
                break
        window["games"] += 1
        window["reward"] += reward
        window["non_loss"] += result != other(mark)
        window["moves"] += board_moves
        if episode % config.history_interval == 0 or episode == config.episodes:
            games = window["games"]
            history.append({"episode": episode, "epsilon": epsilon,
                            "average_reward": window["reward"] / games,
                            "non_loss_rate": window["non_loss"] / games,
                            "average_moves": window["moves"] / games,
                            "average_loss": window["loss"] / max(window["updates"], 1),
                            "updates": agent.updates})
            if progress:
                print(f"{episode:,}/{config.episodes:,} games | "
                      f"reward {history[-1]['average_reward']:+.3f} | "
                      f"non-loss {history[-1]['non_loss_rate']:.1%}")
            window = {"games": 0, "reward": 0.0, "non_loss": 0, "moves": 0,
                      "loss": 0.0, "updates": 0}
    return agent, history


def make_plots(history: list[dict], output: Path) -> None:
    """Save DQN training curves for the lab comparison."""
    output.mkdir(parents=True, exist_ok=True)
    xs = [row["episode"] for row in history]
    for filename, key, title, ylabel, color in (
        ("reward.png", "average_reward", "DQN reward vs episode", "Mean reward", "#2563eb"),
        ("success.png", "non_loss_rate", "DQN non-loss vs episode", "Win + draw rate", "#059669"),
        ("loss.png", "average_loss", "DQN replay loss vs episode", "Huber loss", "#dc2626"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.plot(xs, [row[key] for row in history], color=color, linewidth=2)
        ax.set(title=title, xlabel="Training episode", ylabel=ylabel)
        ax.grid(alpha=0.2)
        fig.savefig(output / filename, dpi=150, facecolor="white")
        plt.close(fig)




# Part 3: train the agent, measure its results, then play against it.
import ipywidgets as widgets
from IPython.display import Image, display
import matplotlib.pyplot as plt
from io import BytesIO

def figure_png(fig):
    # Export plots as PNG so notebook viewers display them reliably.
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=120, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()

EPISODES = 160_000
SEED = 42
# A fixed seed makes the training run reproducible for a lab demonstration.
agent = QLearningAgent(alpha=0.15, gamma=0.97, seed=SEED)
opponent_rng = random.Random(SEED + 1)
# On each opponent turn, choose random/tactical/minimax with these weights.
opponent_weights = (0.25, 0.15, 0.60)  # random, tactical, minimax
history = []
start_names = ("Top-left", "Top", "Top-right", "Left", "Center",
               "Right", "Bottom-left", "Bottom", "Bottom-right")
opening_q_history = []
opening_performance = []
window = {"wins": 0, "draws": 0, "losses": 0, "reward": 0.0,
          "moves": 0, "decisions": 0, "td_error": 0.0}
total_reward = 0.0
q_logs = [widgets.Output(layout=widgets.Layout(max_height="320px", overflow="auto"))
          for _ in range(9)]
training_table = widgets.Output(layout=widgets.Layout(max_height="320px", overflow="auto"))
opening_q_log = widgets.Output(layout=widgets.Layout(max_height="320px", overflow="auto"))

def q_snapshot(episode, start, name):
    # Hold the opening fixed to compare its Q-values at different episodes.
    # "X" marks the human's occupied square; only empty squares are actions.
    board = play(EMPTY_BOARD, start, X)
    values = agent.values(encode_state(board, O))
    best = agent.greedy_actions(board, O)
    lines = [f"Episode {episode:,} | human X at {name}, AI O to move | states: {len(agent.q)}"]
    for row in range(0, 9, 3):
        lines.append("  " + " | ".join("   X   " if board[i] else f"{values[i]:+.3f}"
                                      for i in range(row, row + 3)))
    replies = ", ".join(f"({i // 3 + 1},{i % 3 + 1})" for i in best)
    lines.append(f"  Best AI reply: {replies} | Q = {values[best[0]]:+.3f}\n")
    return "\n".join(lines), values[best[0]]

def opening_checkpoint(episode, games=100):
    # Play 100 test games for each opening with exploration switched off.
    # This measures improvement; test games never call agent.update().
    rates = []
    for start in range(9):
        rng = random.Random(SEED + 5000 + start)
        wins = 0
        for _ in range(games):
            board = play(EMPTY_BOARD, start, X)
            while not terminal(board):
                turn = X if sum(cell != 0 for cell in board) % 2 == 0 else O
                action = (agent.select_action(board, O, rng=rng) if turn == O
                          else mixture_action(board, X, rng, (1, 0, 0)))
                board = play(board, action, turn)
            wins += winner(board) == O
        rates.append(wins / games)
    opening_performance.append({"episode": episode, "win_rates": rates})

for episode in range(1, EPISODES + 1):
    # One episode is one complete game. Alternate X/O to learn both roles.
    agent_mark = X if episode % 2 else O
    board = EMPTY_BOARD
    moves = 0
    decisions = 0
    episode_error = 0.0
    if agent_mark == O:
        board = play(board, mixture_action(board, X, opponent_rng, opponent_weights), X)
        moves += 1
    # Epsilon is the chance of exploring a random legal move.
    # It decreases from 1.0 to 0.03, then stays at 0.03.
    fraction = min(1.0, (episode - 1) / max(1, EPISODES * 0.85))
    epsilon = 1.0 + fraction * (0.03 - 1.0)
    while True:
        # The AI observes a state, acts, then sees the opponent's reply.
        # Its next state is the board when it gets another turn.
        state = encode_state(board, agent_mark)
        action = agent.select_action(board, agent_mark, epsilon)
        board = play(board, action, agent_mark)
        moves += 1
        decisions += 1
        if not terminal(board):
            opponent_mark = other(agent_mark)
            reply = mixture_action(board, opponent_mark, opponent_rng, opponent_weights)
            board = play(board, reply, opponent_mark)
            moves += 1
        done = terminal(board)
        result = winner(board)
        # Intermediate moves earn 0; the game result gives the final reward.
        # AI win = +1, draw = +0.3, AI loss = -1.
        reward = ((1.0 if result == agent_mark else -1.0) if result else 0.3) if done else 0.0
        # Apply Q-learning to this state-action pair after the reply.
        episode_error += agent.update(state, action, reward,
                                      None if done else encode_state(board, agent_mark), done)
        if done:
            break

    window["wins"] += result == agent_mark
    window["draws"] += result == 0
    window["losses"] += result == other(agent_mark)
    window["reward"] += reward
    window["moves"] += moves
    window["decisions"] += decisions
    window["td_error"] += episode_error
    total_reward += reward
    if episode % 1000 == 0 or episode == EPISODES:
        # Summarize the last 1,000 games for the learning curves.
        # Save nine opening Q-tables so the same states can be compared.
        count = episode % 1000 or 1000
        history.append({"episode": episode, "epsilon": epsilon,
                        "avg_reward": window["reward"] / count,
                        "cumulative_reward": total_reward,
                        "non_loss": (window["wins"] + window["draws"]) / count,
                        "avg_moves": window["moves"] / count,
                        "avg_decisions": window["decisions"] / count,
                        "mean_abs_td_error": window["td_error"] / window["decisions"],
                        "wins": window["wins"], "draws": window["draws"],
                        "losses": window["losses"]})
        best_values = []
        for start, name in enumerate(start_names):
            snapshot, best_q = q_snapshot(episode, start, name)
            q_logs[start].append_stdout(snapshot)
            best_values.append(best_q)
        opening_q_history.append({"episode": episode, "best_values": best_values})
        if episode % 10_000 == 0 or episode == EPISODES:
            opening_checkpoint(episode)
        window = {"wins": 0, "draws": 0, "losses": 0, "reward": 0.0,
                  "moves": 0, "decisions": 0, "td_error": 0.0}

table_lines = ["Training results by 1,000-game window",
               f"{'Episode':>8} {'eps':>6} {'W':>5} {'D':>5} {'L':>5} {'avg reward':>11} {'non-loss':>9} {'moves':>7}"]
for row in history:
    table_lines.append(f"{row['episode']:8,d} {row['epsilon']:6.3f} {row['wins']:5d} "
                       f"{row['draws']:5d} {row['losses']:5d} {row['avg_reward']:11.3f} "
                       f"{row['non_loss']:9.1%} {row['avg_moves']:7.2f}")
training_table.append_stdout("\n".join(table_lines) + "\n")
print(f"Training complete: {EPISODES:,} games; final 1,000-game window: "
      f"{history[-1]['non_loss']:.1%} non-loss, {history[-1]['avg_reward']:+.3f} mean reward.")

# Compare the learned greedy policy with three opponents.
# These games only measure performance; they do not update the Q-table.
print("\nFinal evaluation (200 games as X and 200 as O per opponent)")
print(f"{'Opponent':<10} {'Wins':>5} {'Draws':>6} {'Losses':>7} {'Non-loss':>9}")
for name, weights in (("Random", (1, 0, 0)), ("Tactical", (0, 1, 0)),
                      ("Minimax", (0, 0, 1))):
    eval_rng = random.Random(SEED + 100)
    wins = draws = losses = 0
    for game_number in range(400):
        agent_mark = X if game_number % 2 == 0 else O
        board = EMPTY_BOARD
        while not terminal(board):
            turn = X if sum(cell != 0 for cell in board) % 2 == 0 else O
            action = (agent.select_action(board, agent_mark, rng=eval_rng)
                      if turn == agent_mark else mixture_action(board, turn, eval_rng, weights))
            board = play(board, action, turn)
        result = winner(board)
        wins += result == agent_mark
        draws += result == 0
        losses += result == other(agent_mark)
    print(f"{name:<10} {wins:5d} {draws:6d} {losses:7d} {(wins + draws) / 400:9.1%}")

# Keep the AI policy fixed and vary only the human's first square.
# This isolates how the opening position affects the outcome.
OPENING_GAMES = 1000
opening_results = []
print("\nHuman starts as X; AI is O. Each opening gets 1,000 games vs random X replies.")
print("Q is learned future return, not a win probability. X marks the occupied cell.")
for start, name in enumerate(start_names):
    first_board = play(EMPTY_BOARD, start, X)
    q_values = agent.values(encode_state(first_board, O))
    best = agent.greedy_actions(first_board, O)
    category_rng = random.Random(SEED + 1000 + start)
    wins = draws = losses = 0
    for _ in range(OPENING_GAMES):
        board = first_board
        while not terminal(board):
            turn = X if sum(cell != 0 for cell in board) % 2 == 0 else O
            action = (agent.select_action(board, O, rng=category_rng) if turn == O
                      else mixture_action(board, X, category_rng, (1, 0, 0)))
            board = play(board, action, turn)
        result = winner(board)
        wins += result == O
        draws += result == 0
        losses += result == X
    opening_results.append({"start": name, "best_reply": best[0],
                            "best_q": q_values[best[0]], "wins": wins,
                            "draws": draws, "losses": losses})
    q_lines = [f"\n{name}: human X at ({start // 3 + 1},{start % 3 + 1})"]
    for row in range(0, 9, 3):
        q_lines.append("  " + " | ".join("   X   " if first_board[i] else f"{q_values[i]:+.3f}"
                                           for i in range(row, row + 3)))
    q_lines.append(f"  Best AI reply: ({best[0] // 3 + 1},{best[0] % 3 + 1}), "
                   f"Q = {q_values[best[0]]:+.3f}; outcomes: {wins} W / {draws} D / {losses} L\n")
    opening_q_log.append_stdout("\n".join(q_lines))

print("\nResults by human opening (AI perspective; 1,000 games per row)")
print(f"{'Opening':<13} {'AI reply':>8} {'Best Q':>8} {'Wins':>6} {'Draws':>6} "
      f"{'Losses':>6} {'Win rate':>9}")
for row in opening_results:
    action = row["best_reply"]
    print(f"{row['start']:<13} {f'({action // 3 + 1},{action % 3 + 1})':>8} "
          f"{row['best_q']:+8.3f} {row['wins']:6d} {row['draws']:6d} "
          f"{row['losses']:6d} {row['wins'] / OPENING_GAMES:9.1%}")

# Standard RL curves show training reward, success, moves, TD error, and epsilon.
# TD error is the gap between the old Q estimate and its learning target.
xs = [row["episode"] for row in history]
def learning_curve(title, ylabel, series, ylim=None):
    # Plot one training metric against episode number.
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for label, values, color in series:
        ax.plot(xs, values, label=label, color=color, linewidth=2)
    ax.set(title=title, xlabel="Training episode", ylabel=ylabel)
    ax.grid(alpha=0.25)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if len(series) > 1:
        ax.legend()
    display(Image(data=figure_png(fig)))

print("\nSTANDARD RL GRAPHS (1,000-game training windows)")
learning_curve("Reward vs episode", "Mean terminal reward per game",
               [("Mean reward", [r["avg_reward"] for r in history], "#2563eb")])
learning_curve("Success rate vs episode", "Wins + draws / games",
               [("Training non-loss rate", [r["non_loss"] for r in history], "#059669")], (0, 1.05))
learning_curve("Steps per episode vs episode", "Mean moves per game",
               [("Board moves", [r["avg_moves"] for r in history], "#d97706"),
                ("AI decisions", [r["avg_decisions"] for r in history], "#7c3aed")])
learning_curve("TD error vs episode", "Mean absolute TD error / update",
               [("Learning error", [r["mean_abs_td_error"] for r in history], "#dc2626")])
print("Epsilon is one global training schedule; it does not depend on the opening square.")
learning_curve("Global exploration schedule (all openings)", "Epsilon",
               [("Exploration", [r["epsilon"] for r in history], "#0891b2")], (0, 1.05))

fig, axes = plt.subplots(1, 3, figsize=(15, 3.7), constrained_layout=True)
for ax, key, title, ylabel, color in zip(
    axes,
    ("avg_reward", "non_loss", "avg_moves"),
    ("Reward vs episode", "Success rate vs episode", "Steps per episode vs episode"),
    ("Mean terminal reward", "Training non-loss rate", "Mean board moves"),
    ("#2563eb", "#059669", "#d97706"),
):
    ax.plot(xs, [row[key] for row in history], color=color, linewidth=1.8)
    ax.set(title=title, xlabel="Training episode", ylabel=ylabel)
    ax.grid(alpha=0.25)
axes[1].set_ylim(0, 1.05)
print("\nTRAINING PLOTS: reward, success rate, and steps versus episode")
display(Image(data=figure_png(fig)))

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
# Compare final outcomes and best first-reply Q-value by opening.
positions = list(range(9))
win_rates = [row["wins"] / OPENING_GAMES for row in opening_results]
draw_rates = [row["draws"] / OPENING_GAMES for row in opening_results]
loss_rates = [row["losses"] / OPENING_GAMES for row in opening_results]
axes[0].bar(positions, win_rates, label="AI wins", color="#059669")
axes[0].bar(positions, draw_rates, bottom=win_rates, label="Draws", color="#94a3b8")
axes[0].bar(positions, loss_rates, bottom=[w + d for w, d in zip(win_rates, draw_rates)],
            label="AI losses", color="#dc2626")
axes[0].set(title="Outcomes by human first square", ylabel="Share of 1,000 games", ylim=(0, 1.02))
axes[0].legend()
axes[1].bar(positions, [row["best_q"] for row in opening_results], color="#2563eb")
axes[1].set(title="Best learned Q-value by human first square", ylabel="Q-value")
for ax in axes:
    ax.set_xticks(positions, start_names, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.2)
print("\nOPENING GRAPHS: outcomes and best AI Q-value by human first square")
display(Image(data=figure_png(fig)))

fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True, sharey=True,
                         constrained_layout=True)
# Track how the best Q-value changes for each starting square.
for start, ax in enumerate(axes.flat):
    color = "#2563eb" if start in (0, 2, 6, 8) else "#d97706" if start != 4 else "#059669"
    ax.plot([row["episode"] for row in opening_q_history],
            [row["best_values"][start] for row in opening_q_history],
            color=color, linewidth=1.7)
    ax.set_title(start_names[start])
    ax.grid(alpha=0.2)
    if start >= 6:
        ax.set_xlabel("Training episode")
    if start % 3 == 0:
        ax.set_ylabel("Best legal Q")
fig.suptitle("Best AI reply Q-value during training, by human opening")
print("\nQ-VALUE PROGRESS: all nine human starting squares, measured every 1,000 games")
display(Image(data=figure_png(fig)))

fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True, sharey=True,
                         constrained_layout=True)
# Plot greedy win rate for each opening at training checkpoints.
for start, ax in enumerate(axes.flat):
    ax.plot([row["episode"] for row in opening_performance],
            [row["win_rates"][start] for row in opening_performance],
            color="#059669", marker="o", markersize=3, linewidth=1.7)
    ax.set_title(start_names[start])
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.2)
    if start >= 6:
        ax.set_xlabel("Training episode")
    if start % 3 == 0:
        ax.set_ylabel("AI win rate")
fig.suptitle("AI win rate by human first square: 100 greedy games per checkpoint")
print("\nOPENING LEARNING CURVES: AI win rate for each human start, evaluated every 10,000 games")
display(Image(data=figure_png(fig)))

q_tabs = widgets.Tab(children=q_logs)
for start, name in enumerate(start_names):
    q_tabs.set_title(start, name)
details = widgets.Accordion(children=[training_table, q_tabs, opening_q_log], selected_index=None)
for index, title in enumerate(("Full training table", "Q-table every 1,000 games",
                               "Q-tables for each human opening")):
    details.set_title(index, title)
display(details)

def draw_decision_bars(ax, board, values, selected):
    # One bar per square: green = chosen, blue = legal, gray = occupied.
    # Gray bars are plotted at zero because occupied moves cannot be chosen.
    ax.set_facecolor("white")
    colors = ["#cbd5e1" if board[i] else "#16a34a" if i == selected else "#2563eb"
              for i in range(9)]
    heights = [values[i] if board[i] == 0 else 0 for i in range(9)]
    ax.bar(range(1, 10), heights, color=colors, width=0.7)
    ax.axhline(0, color="#334155", linewidth=0.8)
    ax.set_xticks(range(1, 10))
    ax.set(title="AI decision: green = chosen", xlabel="Square (1–9, row by row)", ylabel="Q-value")
    ax.tick_params(labelsize=8, colors="black")
    ax.title.set_color("black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.grid(axis="y", alpha=0.2)
    return ax

def decision_figure(board, values, selected):
    fig, ax = plt.subplots(figsize=(3.25, 2.35), dpi=110)
    fig.patch.set_facecolor("white")
    draw_decision_bars(ax, board, values, selected)
    fig.tight_layout()
    return fig

fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharey=True, constrained_layout=True)
for start, ax in enumerate(axes.flat):
    board = play(EMPTY_BOARD, start, X)
    values = agent.values(encode_state(board, O))
    selected = agent.greedy_actions(board, O)[0]
    draw_decision_bars(ax, board, values, selected)
    ax.set_title(f"{start_names[start]}: AI picks {selected + 1}")
    if start < 6:
        ax.set_xlabel("")
    if start % 3:
        ax.set_ylabel("")
fig.suptitle("AI first decision for each human opening (green = chosen)")
print("\nAI DECISION GRAPHS: all nine human starting squares")
display(Image(data=figure_png(fig)))

# Train a neural DQN on the same board and rewards as the tabular agent.
DQN_EPISODES = 20_000
dqn_agent, dqn_history = train(DQNConfig(episodes=DQN_EPISODES, seed=SEED),
                               progress=False)
print(f"\nDQN training complete: {DQN_EPISODES:,} games; "
      f"final training non-loss {dqn_history[-1]['non_loss_rate']:.1%}")

fig, axes = plt.subplots(1, 3, figsize=(15, 3.7), constrained_layout=True)
for ax, key, title, ylabel, color in zip(
    axes,
    ("average_reward", "non_loss_rate", "average_loss"),
    ("DQN reward", "DQN success rate", "DQN replay loss"),
    ("Mean reward", "Wins + draws / games", "Huber loss"),
    ("#2563eb", "#059669", "#dc2626"),
):
    ax.plot([row["episode"] for row in dqn_history],
            [row[key] for row in dqn_history], color=color, linewidth=2)
    ax.set(title=title, xlabel="DQN training episode", ylabel=ylabel)
    ax.grid(alpha=0.2)
axes[1].set_ylim(0, 1.05)
print("\nDQN TRAINING PLOTS: reward, success rate, and replay loss")
display(Image(data=figure_png(fig)))

def compare_policy(model, weights, seed, games=400):
    # Both agents play the same number of greedy evaluation games.
    rng = random.Random(seed)
    wins = draws = losses = 0
    for game_number in range(games):
        mark = X if game_number % 2 == 0 else O
        board = EMPTY_BOARD
        while not terminal(board):
            turn = X if sum(cell != 0 for cell in board) % 2 == 0 else O
            action = (model.select_action(board, mark, epsilon=0.0, rng=rng)
                      if turn == mark else mixture_action(board, turn, rng, weights))
            board = play(board, action, turn)
        result = winner(board)
        wins += result == mark
        draws += result == 0
        losses += result == other(mark)
    return wins, draws, losses

print("\nGREEDY COMPARISON: 400 games per opponent; Q-table trained 160k, DQN trained 20k")
print(f"{'Agent':<12} {'Opponent':<10} {'Wins':>5} {'Draws':>6} {'Losses':>7}")
comparison = []
for index, (name, weights) in enumerate((("Random", (1, 0, 0)),
                                         ("Tactical", (0, 1, 0)),
                                         ("Minimax", (0, 0, 1)))):
    for label, model in (("Q-table", agent), ("DQN", dqn_agent)):
        wins, draws, losses = compare_policy(model, weights, SEED + 8000 + index)
        comparison.append((label, name, wins, draws, losses))
        print(f"{label:<12} {name:<10} {wins:5d} {draws:6d} {losses:7d}")

fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
positions = np.arange(len(comparison))
win_rates = [row[2] / 400 for row in comparison]
draw_rates = [row[3] / 400 for row in comparison]
loss_rates = [row[4] / 400 for row in comparison]
ax.bar(positions, win_rates, label="Wins", color="#16a34a")
ax.bar(positions, draw_rates, bottom=win_rates, label="Draws", color="#94a3b8")
ax.bar(positions, loss_rates,
       bottom=[w + d for w, d in zip(win_rates, draw_rates)],
       label="Losses", color="#dc2626")
ax.set_xticks(positions, [f"{row[1]}\n{row[0]}" for row in comparison])
ax.set(title="Greedy game outcomes (different training budgets)",
       ylabel="Share of 400 games", ylim=(0, 1.05))
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.2)
print("\nQ-LEARNING VS DQN: wins, draws, and losses; training budgets differ")
display(Image(data=figure_png(fig)))


class NotebookTicTacToe:
    """Let a human play while showing why the trained AI chose its move."""
    def __init__(self, tabular_agent, neural_agent):
        self.agents = {"Q-table": tabular_agent, "DQN": neural_agent}
        self.model_name = "Q-table"
        self.agent = tabular_agent
        self.board = EMPTY_BOARD
        self.human_mark = X
        self.agent_mark = O
        self.finished = False

        self.title = widgets.Label(value="TIC TAC TOE  ·  Q-LEARNING / DQN")
        self.status = widgets.Label()
        self.tabular_button = widgets.Button(description="Q-table", layout=widgets.Layout(width="120px"))
        self.dqn_button = widgets.Button(description="DQN", layout=widgets.Layout(width="120px"))
        self.tabular_button.on_click(lambda _: self._change_model("Q-table"))
        self.dqn_button.on_click(lambda _: self._change_model("DQN"))
        self.model_choice = widgets.HBox([self.tabular_button, self.dqn_button])
        self.play_x = widgets.Button(description="Play as X", layout=widgets.Layout(width="120px"))
        self.play_o = widgets.Button(description="Play as O", layout=widgets.Layout(width="120px"))
        self.play_x.on_click(lambda _: self._change_side(X))
        self.play_o.on_click(lambda _: self._change_side(O))
        self.side = widgets.HBox([self.play_x, self.play_o])
        self.new_game = widgets.Button(description="New game", icon="refresh")
        self.new_game.on_click(lambda _: self.reset())

        self.cells = []
        for position in range(9):
            button = widgets.Button(
                description="", layout=widgets.Layout(width="82px", height="82px"),
            )
            button.style.font_weight = "bold"
            button.style.font_size = "34px"
            button.on_click(lambda _, index=position: self.human_move(index))
            self.cells.append(button)

        self.q_title = widgets.Label(value="Q-values at the AI's decision")
        self.q_state = widgets.Label(value="Make a move to see the Q-table.")
        self.q_choice = widgets.Label(value="0 = empty · 1 = AI · 2 = you")
        self.decision_chart = widgets.Image(format="png",
                                            layout=widgets.Layout(width="330px", display="none"))
        self.q_cells = [widgets.Button(
            description="—", layout=widgets.Layout(width="92px", height="52px"),
        ) for _ in range(9)]
        for button in self.q_cells:
            button.style.font_size = "14px"
            button.style.font_weight = "bold"
            button.tooltip = "Q value at the AI's last decision"

        # Notebook dark themes can override ButtonStyle.text_color.
        self.font_fix = widgets.HTML(value="""<style>
        .rl-black-text, .rl-black-text *, button.rl-black-text {
            color: #000 !important;
            -webkit-text-fill-color: #000 !important;
        }
        </style>""")
        for button in (self.tabular_button, self.dqn_button, self.play_x, self.play_o,
                       self.new_game, *self.cells, *self.q_cells):
            button.add_class("rl-black-text")
            button.style.text_color = "#000000"

        board_grid = widgets.GridBox(
            self.cells,
            layout=widgets.Layout(grid_template_columns="repeat(3, 82px)", grid_gap="5px"),
        )
        q_grid = widgets.GridBox(
            self.q_cells,
            layout=widgets.Layout(grid_template_columns="repeat(3, 92px)", grid_gap="5px"),
        )
        left = widgets.VBox([self.font_fix, self.title, self.model_choice,
                             self.side, self.status, board_grid, self.new_game])
        right = widgets.VBox([self.q_title, self.q_state, q_grid,
                              self.q_choice, self.decision_chart])
        self.widget = widgets.HBox(
            [left, right],
            layout=widgets.Layout(flex_flow="row wrap", gap="32px", align_items="flex-start"),
        )
        self.reset()

    def _change_side(self, mark):
        self.human_mark = mark
        self.agent_mark = other(mark)
        self.reset()

    def _change_model(self, name):
        self.model_name = name
        self.agent = self.agents[name]
        self.reset()

    def _draw_model_choice(self):
        for button, name in ((self.tabular_button, "Q-table"),
                             (self.dqn_button, "DQN")):
            button.style.button_color = "#50d99a" if name == self.model_name else "#f8fafc"
            button.style.text_color = "#000000"
            button.style.font_weight = "bold"

    def _draw_side(self):
        for button, mark in ((self.play_x, X), (self.play_o, O)):
            chosen = mark == self.human_mark
            button.style.button_color = "#50d99a" if chosen else "#f8fafc"
            button.style.text_color = "#000000"
            button.style.font_weight = "bold"

    def _draw_board(self):
        # Redraw X/O marks; occupied squares remain visible but cannot be played.
        for index, button in enumerate(self.cells):
            mark = self.board[index]
            button.description = "X" if mark == X else "O" if mark == O else ""
            button.style.button_color = "#72e0a6" if mark == X else "#ffbe73" if mark == O else "#f8fafc"
            button.style.text_color = "#000000"
            button.tooltip = "Empty square" if mark == 0 else "Occupied square"

    def _finish_or_continue(self):
        # End on a win/draw; otherwise return control to the human.
        result = winner(self.board)
        if result or terminal(self.board):
            self.finished = True
            self.status.value = (
                "You win!" if result == self.human_mark else
                "AI wins!" if result == self.agent_mark else "Draw!"
            )
        else:
            self.status.value = "Your turn. Choose an empty square."
        self._draw_board()

    def reset(self):
        # Clear the board and hide the chart until the next AI move.
        self.board = EMPTY_BOARD
        self.finished = False
        self._draw_model_choice()
        self._draw_side()
        self.q_title.value = ("Q-values from the " + self.model_name + " at the AI's decision")
        self.q_state.value = "Make a move to see the AI's values."
        self.q_choice.value = "0 = empty · 1 = AI · 2 = you"
        self.decision_chart.value = b""
        self.decision_chart.layout.display = "none"
        for button in self.q_cells:
            button.description = "—"
            button.style.button_color = "#f8fafc"
            button.style.text_color = "#000000"
        self.status.value = "Your turn. Choose an empty square."
        self._draw_board()
        if self.agent_mark == X:
            self._agent_move()

    def human_move(self, index):
        # Validate the click, place the human mark, then let the AI reply.
        if self.finished or self.board[index] != 0:
            return
        self.board = play(self.board, index, self.human_mark)
        if terminal(self.board):
            self._finish_or_continue()
        else:
            self._agent_move()

    def _agent_move(self):
        # Inference only: read Q-values and choose a best legal action.
        # epsilon=0 means no random exploration during the demonstration.
        before = self.board
        state = encode_state(before, self.agent_mark)
        values = self.agent.values(state)
        selected = self.agent.select_action(before, self.agent_mark, epsilon=0.0)
        self.q_state.value = "State: " + state + "  (AI's view)"
        # Display the values from before the AI marks its selected square.
        for index, button in enumerate(self.q_cells):
            button.description = "occupied" if before[index] != 0 else f"{values[index]:+.3f}"
            button.style.button_color = (
                "#72e0a6" if index == selected else
                "#dbe4ea" if before[index] != 0 else "#f8fafc"
            )
            button.style.text_color = "#000000"
        row, col = divmod(selected, 3)
        self.q_choice.value = (
            f"AI chose row {row + 1}, column {col + 1} · Q = {values[selected]:+.3f}"
        )
        fig = decision_figure(before, values, selected)
        self.decision_chart.value = figure_png(fig)
        self.decision_chart.layout.display = "block"
        self.board = play(before, selected, self.agent_mark)
        self._finish_or_continue()


game = NotebookTicTacToe(agent, dqn_agent)
display(game.widget)
