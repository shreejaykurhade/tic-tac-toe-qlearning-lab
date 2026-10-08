"""Learn one expected reward for each board state and legal move."""

import json
from pathlib import Path
import random

from .core import Board, canonicalize, encode_state, legal_actions, transform_state, SYMMETRIES


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

    def export(self, path: str | Path, metadata: dict | None = None) -> dict:
        """Expand observed canonical states to raw coordinates for browser inference.

        Symmetric boards can have multiple valid coordinate mappings. We call the
        same deterministic canonicalization used by training for each raw state;
        this preserves exactly the trained policy, including symmetric ties.
        """
        raw_states = {transform_state(state, p) for state in self.q for p in SYMMETRIES}
        table = {state: self.values(state) for state in sorted(raw_states)}
        payload = {
            "metadata": {"algorithm": "tabular Q-learning", "state_encoding": "0 empty, 1 agent, 2 opponent",
                         "action_encoding": "row-major indices 0..8", "symmetry_reduction": "8 rotations/reflections",
                         "canonical_states": len(self.q), "exported_states": len(table),
                         "greedy_tie_tolerance": 1e-12, "unseen_state_values": 0.0,
                         "alpha": self.alpha, "gamma": self.gamma, **(metadata or {})},
            "q_table": table,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        return payload

    @classmethod
    def load(cls, path: str | Path, seed: int = 42) -> "QLearningAgent":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        metadata = payload["metadata"]
        agent = cls(metadata.get("alpha", 0.15), metadata.get("gamma", 0.97), seed)
        for state, values in payload["q_table"].items():
            key, permutation = canonicalize(state)
            if key == state:
                agent.q[key] = [0.0] * 9
                for action in range(9):
                    agent.q[key][permutation[action]] = values[action]
        return agent
