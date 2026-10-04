"""Reproducible training/evaluation CLI: python -m tictactoe.train."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import platform
import random
import time

from .agent import QLearningAgent
from .core import EMPTY_BOARD, X, O, encode_state, mixture_action, other, play, terminal, winner
from .evaluate import adversarial_audit, evaluate


@dataclass
class TrainingConfig:
    episodes: int = 160_000
    seed: int = 42
    alpha: float = 0.15
    gamma: float = 0.97
    epsilon_start: float = 1.0
    epsilon_end: float = 0.03
    epsilon_decay_fraction: float = 0.85
    win_reward: float = 1.0
    draw_reward: float = 0.3
    loss_reward: float = -1.0
    random_weight: float = 0.25
    tactical_weight: float = 0.15
    minimax_weight: float = 0.60
    history_interval: int = 1000
    evaluation_interval: int = 5000
    checkpoint_games_per_role: int = 200
    final_games_per_role: int = 2000
    evaluation_seed: int = 2026


def epsilon_at(episode: int, config: TrainingConfig) -> float:
    """Linear annealing followed by a nonzero exploration floor."""
    fraction = min(1.0, episode / max(1., config.episodes * config.epsilon_decay_fraction))
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def train(config: TrainingConfig, progress: bool = True):
    if config.episodes < 1 or config.history_interval < 1 or config.evaluation_interval < 1:
        raise ValueError("Episode counts and intervals must be positive.")
    started = time.perf_counter()
    agent = QLearningAgent(config.alpha, config.gamma, config.seed)
    environment_rng = random.Random(config.seed + 1)
    history, checkpoints = [], []
    block = {"episodes": 0, "wins": 0, "draws": 0, "losses": 0, "reward": 0., "moves": 0, "decisions": 0, "td_error": 0.}
    total_reward = 0.
    weights = (config.random_weight, config.tactical_weight, config.minimax_weight)
    for episode in range(1, config.episodes + 1):
        mark = X if episode % 2 else O
        board, moves, decisions, error = EMPTY_BOARD, 0, 0, 0.
        if mark == O:
            board = play(board, mixture_action(board, X, environment_rng, weights), X)
            moves += 1
        epsilon = epsilon_at(episode - 1, config)
        while True:
            state = encode_state(board, mark)
            action = agent.select_action(board, mark, epsilon)
            board = play(board, action, mark)
            moves += 1
            decisions += 1
            if not terminal(board):
                response = mixture_action(board, other(mark), environment_rng, weights)
                board = play(board, response, other(mark))
                moves += 1
            done = terminal(board)
            won = winner(board)
            reward = ((config.win_reward if won == mark else config.loss_reward) if won else config.draw_reward) if done else 0.
            error += agent.update(state, action, reward, None if done else encode_state(board, mark), done)
            if done:
                break
        block["episodes"] += 1
        block["wins"] += won == mark
        block["draws"] += won == 0
        block["losses"] += won == other(mark)
        block["reward"] += reward
        block["moves"] += moves
        block["decisions"] += decisions
        block["td_error"] += error
        total_reward += reward
        if episode % config.history_interval == 0 or episode == config.episodes:
            count = block["episodes"]
            history.append({"episode": episode, "window_episodes": count, "epsilon": epsilon,
                            "average_reward": block["reward"] / count, "cumulative_reward": total_reward,
                            "win_rate": block["wins"] / count, "draw_rate": block["draws"] / count,
                            "loss_rate": block["losses"] / count,
                            "non_loss_rate": (block["wins"] + block["draws"]) / count,
                            "average_board_moves": block["moves"] / count,
                            "average_agent_decisions": block["decisions"] / count,
                            "mean_absolute_td_error": block["td_error"] / block["decisions"],
                            "canonical_states": len(agent.q)})
            block = {key: 0 for key in block}
        if episode % config.evaluation_interval == 0 or episode == config.episodes:
            measurements = evaluate(agent, config.checkpoint_games_per_role, config.evaluation_seed,
                                    config.draw_reward, config.loss_reward)
            for name, values in measurements.items():
                checkpoints.append({"episode": episode, "opponent": name, **values["combined"]})
            if progress:
                values = " | ".join(f"{name}: {v['combined']['non_loss_rate']:.1%} non-loss" for name, v in measurements.items())
                print(f"{episode:,}/{config.episodes:,} episodes | epsilon={epsilon:.3f} | {values}", flush=True)
    return agent, history, checkpoints, time.perf_counter() - started


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(history: list[dict], checkpoints: list[dict], output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#f8fafc", "axes.facecolor": "#ffffff"})
    output.mkdir(exist_ok=True)
    xs = [row["episode"] for row in history]
    def save(name, title, ylabel, series, ylim=None):
        fig, ax = plt.subplots(figsize=(8.2, 4.3), constrained_layout=True)
        colors = ("#2563eb", "#e07025", "#0d9488")
        for (label, xx, yy), color in zip(series, colors):
            ax.plot(xx, yy, label=label, color=color, linewidth=2)
        ax.set(title=title, xlabel="Training episode", ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.ticklabel_format(axis="x", style="plain")
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(frameon=False, loc="best")
        fig.savefig(output / name, dpi=170)
        plt.close(fig)
    save("reward_vs_episode.png", "Training reward improves as exploration falls", "Mean terminal reward / episode",
         [("Training window (1,000 episodes)", xs, [r["average_reward"] for r in history])])
    save("success_vs_episode.png", "Held-out greedy evaluation: non-loss rate", "Wins + draws / games",
         [(name.title(), [r["episode"] for r in checkpoints if r["opponent"] == name],
           [r["non_loss_rate"] for r in checkpoints if r["opponent"] == name]) for name in ("random", "tactical", "minimax")], (0, 1.03))
    save("steps_vs_episode.png", "Episode length during training", "Mean number of moves",
         [("Board moves (both players)", xs, [r["average_board_moves"] for r in history]),
          ("Agent decisions", xs, [r["average_agent_decisions"] for r in history])])
    save("td_error_vs_episode.png", "Temporal-difference error under a stochastic opponent", "Mean absolute TD error / update",
         [("Training window (1,000 episodes)", xs, [r["mean_absolute_td_error"] for r in history])])


def run_experiment(config: TrainingConfig, output_dir: str | Path = "artifacts", plots: bool = True,
                   progress: bool = True) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    baseline = evaluate(QLearningAgent(seed=config.seed), config.final_games_per_role,
                        config.evaluation_seed + 1_000_000, config.draw_reward, config.loss_reward)
    agent, history, checkpoints, elapsed = train(config, progress)
    final = evaluate(agent, config.final_games_per_role, config.evaluation_seed + 1_000_000,
                     config.draw_reward, config.loss_reward)
    audit = adversarial_audit(agent)
    config_dict = asdict(config)
    payload = agent.export(output / "q_table.json", {"seed": config.seed, "episodes": config.episodes,
                          "rewards": {"win": config.win_reward, "draw": config.draw_reward, "loss": config.loss_reward, "ongoing": 0},
                          "training_opponents": {"random": config.random_weight, "tactical": config.tactical_weight, "minimax": config.minimax_weight}})
    evaluation = {"evaluation_seed": config.evaluation_seed + 1_000_000,
                  "games_per_role_per_opponent": config.final_games_per_role,
                  "policy": "epsilon=0, uniformly random maximal-Q ties; no search at inference",
                  "baseline": baseline, "trained": final, "adversarial_audit": audit}
    summary = {"parameters": config_dict, "training_elapsed_seconds": elapsed,
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "python_version": platform.python_version(), "platform": platform.platform(),
               "canonical_states": payload["metadata"]["canonical_states"],
               "exported_states": payload["metadata"]["exported_states"],
               "final_training_window": history[-1], "initial_training_window": history[0],
               "evaluation": evaluation,
               "methodology": {"state": "9 cells relative to agent (0 empty, 1 self, 2 opponent)",
                               "transition": "agent action plus opponent response; next state belongs to same agent",
                               "symmetry": "8 rotations/reflections canonicalized during learning; raw states expanded for browser",
                               "opponent_mixture": "policy sampled independently for each opponent move",
                               "roles": "agent alternates X and O each episode",
                               "checkpoint_seed": config.evaluation_seed,
                               "final_seed": config.evaluation_seed + 1_000_000,
                               "baseline": "all Q-values zero, epsilon=0, uniform legal ties (random policy)",
                               "success_definition": "non-loss rate = (wins + draws) / games"}}
    if plots:
        make_plots(history, checkpoints, output / "plots")
        import matplotlib
        import numpy
        summary["library_versions"] = {"numpy": numpy.__version__, "matplotlib": matplotlib.__version__}
    write_csv(output / "training_history.csv", history)
    write_csv(output / "evaluation_checkpoints.csv", checkpoints)
    (output / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if progress:
        print(json.dumps({"training_seconds": round(elapsed, 2), "canonical_states": len(agent.q),
                          "final": {k: v["combined"] for k, v in final.items()}, "adversarial_audit": audit}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=160_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--evaluation-games", type=int, default=2000, help="Games per role and opponent in final evaluation")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    run_experiment(TrainingConfig(episodes=args.episodes, seed=args.seed, final_games_per_role=args.evaluation_games), args.output, not args.no_plots)


if __name__ == "__main__":
    main()
