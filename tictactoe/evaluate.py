"""Held-out evaluation and an exact adversarial audit of the learned policy."""

from functools import lru_cache
import random

from .agent import QLearningAgent
from .core import EMPTY_BOARD, X, O, Board, OPPONENTS, encode_state, legal_actions, other, play, terminal, winner


def evaluate(agent: QLearningAgent, games_per_role: int = 1000, seed: int = 2026,
             draw_reward: float = 0.3, loss_reward: float = -1.0) -> dict:
    """Evaluate with fresh RNGs; epsilon=0; tie choices remain random."""
    results = {}
    for index, (name, opponent) in enumerate(OPPONENTS.items()):
        role_results = {}
        for mark, role in ((X, "X"), (O, "O")):
            rng = random.Random(seed + 10000 * index + 100 * mark)
            wins = draws = losses = total_moves = 0
            for _ in range(games_per_role):
                board, turn, moves = EMPTY_BOARD, X, 0
                while not terminal(board):
                    action = agent.select_action(board, mark, epsilon=0.0, rng=rng) if turn == mark else opponent(board, turn, rng)
                    board = play(board, action, turn)
                    moves += 1
                    turn = other(turn)
                won = winner(board)
                wins += won == mark
                losses += won == other(mark)
                draws += won == 0
                total_moves += moves
            role_results[role] = _metrics(wins, draws, losses, total_moves, draw_reward, loss_reward)
        role_results["combined"] = _metrics(
            sum(v["wins"] for v in role_results.values()),
            sum(v["draws"] for v in role_results.values()),
            sum(v["losses"] for v in role_results.values()),
            sum(v["total_board_moves"] for v in role_results.values()), draw_reward, loss_reward)
        results[name] = role_results
    return results


def _metrics(wins, draws, losses, total_moves, draw_reward, loss_reward):
    games = wins + draws + losses
    reward = wins + draw_reward * draws + loss_reward * losses
    return {"games": games, "wins": wins, "draws": draws, "losses": losses,
            "win_rate": wins / games, "draw_rate": draws / games, "loss_rate": losses / games,
            "non_loss_rate": (wins + draws) / games, "cumulative_reward": reward,
            "average_reward": reward / games, "total_board_moves": total_moves,
            "average_board_moves": total_moves / games}


def adversarial_audit(agent: QLearningAgent) -> dict:
    """Exactly evaluate an opponent that maximizes this agent's loss probability.

    Unlike ordinary minimax-versus-minimax optimal actions, this adversary may
    choose traps to exploit a fallible learned policy. Agent maximal-Q ties are
    averaged uniformly; opponent choices maximize loss, then minimize wins.
    Returns exact tree probabilities from an empty board for each playing role.
    """
    result = {}
    for agent_mark, role in ((X, "X"), (O, "O")):
        seen_states: set[str] = set()
        unknown_states: set[str] = set()

        @lru_cache(maxsize=None)
        def visit(board: Board, turn: int) -> tuple[float, float, float]:
            won = winner(board)
            if won:
                return (1., 0., 0.) if won == agent_mark else (0., 0., 1.)
            if 0 not in board:
                return (0., 1., 0.)
            if turn == agent_mark:
                state = encode_state(board, agent_mark)
                seen_states.add(state)
                from .core import canonicalize
                if canonicalize(state)[0] not in agent.q:
                    unknown_states.add(state)
                choices = agent.greedy_actions(board, agent_mark)
                children = [visit(play(board, action, turn), other(turn)) for action in choices]
                return tuple(sum(child[i] for child in children) / len(children) for i in range(3))
            children = [visit(play(board, action, turn), other(turn)) for action in legal_actions(board)]
            return max(children, key=lambda scores: (scores[2], -scores[0]))

        win, draw, loss = visit(EMPTY_BOARD, X)
        result[role] = {"win_probability": win, "draw_probability": draw, "loss_probability": loss,
                        "reachable_agent_states": len(seen_states), "unseen_agent_states": len(unknown_states),
                        "tree_nodes": visit.cache_info().currsize,
                        "unbeatable_from_empty_board": loss < 1e-12}
    return result
