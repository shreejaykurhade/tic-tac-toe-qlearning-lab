"""Meaningful rules, TD-update, export, determinism, and opponent checks."""

import json
import random
from pathlib import Path
import tempfile
import unittest

from tictactoe import EMPTY_BOARD, QLearningAgent, X, O, encode_state, legal_actions, play, winner
from tictactoe.core import WIN_LINES, canonicalize, minimax_actions, minimax_value, SYMMETRIES, transform_state
from tictactoe.evaluate import adversarial_audit
from tictactoe.train import TrainingConfig, train


class RulesTests(unittest.TestCase):
    def test_all_eight_winning_lines_for_both_players(self):
        for mark in (X, O):
            for line in WIN_LINES:
                board = tuple(mark if i in line else 0 for i in range(9))
                self.assertEqual(winner(board), mark)
                self.assertEqual(legal_actions(board), [])

    def test_draw_and_legal_action_mask(self):
        draw = (1, 2, 1, 1, 2, 2, 2, 1, 1)
        self.assertEqual(winner(draw), 0)
        self.assertEqual(legal_actions(draw), [])
        with self.assertRaises(ValueError):
            play(draw, 0, X)
        with self.assertRaises(ValueError):
            play((1, 0, 0, 0, 0, 0, 0, 0, 0), 0, O)

    def test_perspective_and_symmetry(self):
        board = (1, 0, 2, 0, 1, 0, 2, 0, 0)
        self.assertEqual(encode_state(board, X), "102010200")
        self.assertEqual(encode_state(board, O), "201020100")
        key = canonicalize(encode_state(board, X))[0]
        self.assertEqual(len(set(SYMMETRIES)), 8)
        for permutation in SYMMETRIES:
            self.assertEqual(canonicalize(transform_state(encode_state(board, X), permutation))[0], key)

    def test_minimax_empty_board_draw_and_forced_block(self):
        self.assertEqual(minimax_value(EMPTY_BOARD, X), 0)
        self.assertEqual(minimax_actions((1, 1, 0, 0, 2, 0, 0, 0, 0), O), (2,))


class LearningTests(unittest.TestCase):
    def test_terminal_update_never_bootstraps(self):
        agent = QLearningAgent(alpha=.5, gamma=.9)
        agent.q["000000000"] = [100.] * 9
        state = "110220000"
        agent.update(state, 2, 1., "000000000", done=True)
        self.assertAlmostEqual(agent.values(state)[2], .5)

    def test_nonterminal_update_masks_illegal_next_actions(self):
        agent = QLearningAgent(alpha=.5, gamma=.9)
        next_state = "120000000"
        key, p = canonicalize(next_state)
        agent.q[key] = [0.] * 9
        agent.q[key][p[0]] = 999.  # occupied: must never enter max target
        agent.q[key][p[1]] = 999.
        agent.q[key][p[2]] = .8
        agent.update("000000000", 0, 0., next_state, done=False)
        self.assertAlmostEqual(agent.values("000000000")[0], .36)

    def test_policy_masks_occupied_actions_and_randomizes_ties(self):
        agent = QLearningAgent(seed=12)
        board = (1, 2, 0, 0, 0, 0, 0, 0, 0)
        key, p = canonicalize(encode_state(board, X))
        agent.q[key] = [0.] * 9
        agent.q[key][p[0]] = 999.
        self.assertEqual(set(agent.select_action(board, X) for _ in range(300)), set(range(2, 9)))

    def test_export_and_reload_preserve_values_in_every_orientation(self):
        agent = QLearningAgent(alpha=.4)
        agent.update("102000000", 4, .6, None, True)
        agent.update("000000000", 0, 1., None, True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q.json"
            payload = agent.export(path)
            restored = QLearningAgent.load(path)
            self.assertEqual(set(payload), {"metadata", "q_table"})
            for state, exported_values in payload["q_table"].items():
                self.assertEqual(agent.values(state), exported_values)
                self.assertEqual(restored.values(state), exported_values)

    def test_training_is_reproducible_and_evaluation_is_nonmutating(self):
        config = TrainingConfig(episodes=300, history_interval=100, evaluation_interval=300,
                                checkpoint_games_per_role=10, final_games_per_role=10)
        one, history_one, checkpoints_one, _ = train(config, progress=False)
        two, history_two, checkpoints_two, _ = train(config, progress=False)
        self.assertEqual(one.q, two.q)
        self.assertEqual(history_one, history_two)
        self.assertEqual(checkpoints_one, checkpoints_two)
        before = json.dumps(one.q, sort_keys=True)
        audit = adversarial_audit(one)
        self.assertEqual(before, json.dumps(one.q, sort_keys=True))
        for result in audit.values():
            self.assertAlmostEqual(sum(result[k] for k in ("win_probability", "draw_probability", "loss_probability")), 1.)


if __name__ == "__main__":
    unittest.main()
