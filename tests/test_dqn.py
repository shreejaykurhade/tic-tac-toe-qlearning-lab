"""Behavior checks for the NumPy DQN."""

import unittest

from tictactoe.core import EMPTY_BOARD, X, O, play
from tictactoe.dqn import DQNAgent, state_vector


class DQNTests(unittest.TestCase):
    def test_state_uses_separate_agent_and_opponent_inputs(self):
        values = state_vector("120000000")
        self.assertEqual(len(values), 18)
        self.assertEqual(values[0], 1)
        self.assertEqual(values[10], 1)
        self.assertEqual(values.sum(), 2)

    def test_greedy_action_never_uses_occupied_square(self):
        agent = DQNAgent(seed=7)
        board = play(EMPTY_BOARD, 0, X)
        for _ in range(20):
            self.assertNotEqual(agent.select_action(board, O), 0)

    def test_terminal_reward_changes_action_value(self):
        agent = DQNAgent(seed=7, batch_size=8, target_interval=1)
        state = "110220000"
        before = agent.values(state)[2]
        for _ in range(32):
            agent.remember(state, 2, 1.0, None, True)
        for _ in range(20):
            loss = agent.train_step()
            self.assertIsNotNone(loss)
        self.assertGreater(agent.values(state)[2], before)
        self.assertGreater(agent.updates, 0)


if __name__ == "__main__":
    unittest.main()
