"""Train and compare the neural DQN with the tabular Q-learning agent."""

from dataclasses import dataclass
import json
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np

from .agent import QLearningAgent
from .core import EMPTY_BOARD, X, O, encode_state, mixture_action, other, play, terminal, winner
from .dqn import DQNAgent
from .evaluate import evaluate


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


def run(config: DQNConfig = DQNConfig(), output: str | Path = "artifacts/dqn") -> dict:
    """Train, evaluate, save model/plots, and compare with the saved Q-table."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    agent, history = train(config)
    dqn_results = evaluate(agent, config.evaluation_games_per_role, seed=2026)
    q_path = Path(__file__).resolve().parents[1] / "artifacts/q_table.json"
    tabular_results = evaluate(QLearningAgent.load(q_path),
                               config.evaluation_games_per_role, seed=2026)
    make_plots(history, output / "plots")
    np.savez(output / "dqn_weights.npz", **agent.online)
    result = {"episodes": config.episodes, "seed": config.seed,
              "evaluation_games_per_role": config.evaluation_games_per_role,
              "dqn": dqn_results, "tabular_q_learning": tabular_results,
              "history": history}
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    run()
