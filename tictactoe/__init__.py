"""Small, dependency-free Tic-Tac-Toe environment and tabular Q-learning agent."""

from .core import Board, EMPTY_BOARD, X, O, legal_actions, play, winner, encode_state
from .agent import QLearningAgent

__all__ = ["Board", "EMPTY_BOARD", "X", "O", "legal_actions", "play", "winner", "encode_state", "QLearningAgent"]
