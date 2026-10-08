"""Game rules and opponents. Cells: 0 empty, 1 X, 2 O."""

from functools import lru_cache
import random
from typing import TypeAlias

Board: TypeAlias = tuple[int, ...]
X, O = 1, 2
EMPTY_BOARD: Board = (0,) * 9
WIN_LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))


def legal_actions(board: Board) -> list[int]:
    """Return legal empty squares."""
    return [] if winner(board) else [i for i, mark in enumerate(board) if mark == 0]


@lru_cache(maxsize=None)
def winner(board: Board) -> int:
    for a, b, c in WIN_LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return 0


def terminal(board: Board) -> bool:
    return bool(winner(board)) or 0 not in board


def play(board: Board, action: int, mark: int) -> Board:
    if len(board) != 9 or mark not in (X, O):
        raise ValueError("Expected a nine-cell board and mark X=1 or O=2.")
    if terminal(board):
        raise ValueError("Cannot play after a terminal state.")
    if action not in range(9) or board[action] != 0:
        raise ValueError("Action must select an empty cell in 0..8.")
    return board[:action] + (mark,) + board[action + 1:]


def encode_state(board: Board, agent_mark: int) -> str:
    """Encode 0 empty, 1 agent, 2 opponent."""
    # The same encoding works whether the agent plays X or O.
    return "".join("0" if m == 0 else "1" if m == agent_mark else "2" for m in board)


def other(mark: int) -> int:
    return 3 - mark


def _symmetries() -> tuple[tuple[int, ...], ...]:
    # Rotate/reflect cell indices.
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
    result = ["0"] * 9
    for i, j in enumerate(permutation):
        result[j] = state[i]
    return "".join(result)


@lru_cache(maxsize=None)
def canonicalize(state: str) -> tuple[str, tuple[int, ...]]:
    """Use one key for symmetric boards."""
    # Return the smallest rotated/reflected state and its move mapping.
    return min(((transform_state(state, p), p) for p in SYMMETRIES), key=lambda pair: pair[0])


@lru_cache(maxsize=None)
def minimax_value(board: Board, to_move: int) -> int:
    """Score perfect play for the current player."""
    won = winner(board)
    if won:
        return 1 if won == to_move else -1
    actions = legal_actions(board)
    if not actions:
        return 0
    return max(-minimax_value(play(board, a, to_move), other(to_move)) for a in actions)


@lru_cache(maxsize=None)
def minimax_actions(board: Board, mark: int) -> tuple[int, ...]:
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
    """Win, block, then prefer center or corners."""
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
    """Choose a training opponent each turn."""
    name = rng.choices(tuple(OPPONENTS), weights=weights, k=1)[0]
    return OPPONENTS[name](board, mark, rng)
