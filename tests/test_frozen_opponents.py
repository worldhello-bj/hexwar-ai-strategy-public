"""Evaluation-only frozen opponents retain the terminal/public-input contract."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from engine.game import Game
from env.episode_api import TwoPlayerEpisode
from env.public_observation import PublicObservationAdapter
from env.match_eval import write_synthetic_snapshot
from env.neural_policy import NeuralPolicy
from env.neural_train import NeuralTrainer
from tools.evaluate_frozen_opponents import FrozenSampler, evaluate, run_game, state_hash


class FrozenOpponentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.snapshot = write_synthetic_snapshot(self.root / 'snapshot')

    def checkpoint(self, schema=2, max_steps=20):
        trainer = NeuralTrainer(self.snapshot, observation_schema=schema,
            policy_config={'buckets': 16, 'hidden_dim': 4},
            episode_config={'max_seconds': 20, 'max_steps': max_steps}, heldout_seeds=[90])
        path = self.root / ('checkpoint-%s-%s.json' % (schema, max_steps))
        trainer.save(path)
        return path

    def test_reproducible_legal_independent_sampling_and_frozen_state(self):
        policy = NeuralPolicy(buckets=16, hidden_dim=4)
        before = state_hash(policy)
        candidates = [{'type': 'x'}, {'type': 'y'}]
        left, right = FrozenSampler(policy, 42), FrozenSampler(policy, 42)
        sequence = [left({}, candidates) for _ in range(30)]
        self.assertEqual(sequence, [right({}, candidates) for _ in range(30)])
        self.assertTrue(all(a in candidates for a in sequence))
        self.assertIsNot(left.rng, right.rng)
        self.assertEqual(state_hash(policy), before)
        sequence[0]['type'] = 'mutated'
        self.assertEqual(candidates, [{'type': 'x'}, {'type': 'y'}])

    def test_self_play_reproducible_paired_and_checkpoint_untouched(self):
        path = self.checkpoint()
        before = path.read_bytes()
        with patch.object(NeuralPolicy, 'sample', side_effect=AssertionError('training RNG used')), \
                patch.object(NeuralPolicy, 'update_batch', side_effect=AssertionError('trained')):
            report = evaluate(path, path, self.snapshot, seeds=[100, 101])
            self.assertEqual(report, evaluate(path, path, self.snapshot, seeds=[100, 101]))
        self.assertEqual(before, path.read_bytes())
        self.assertEqual(report['models']['A']['state_sha256'], report['models']['B']['state_sha256'])
        self.assertEqual([(g['seed'], g['a_side']) for g in report['games']],
                         [(100, 'BLUE'), (100, 'RED'), (101, 'BLUE'), (101, 'RED')])
        # Symmetry means equal exposure and score bookkeeping, not exactly 50% wins.
        self.assertEqual(report['by_a_seat']['BLUE']['games'], report['by_a_seat']['RED']['games'])
        for game in report['games']:
            self.assertEqual(game['a_score_diff'], game['a_score'] - game['b_score'])
            self.assertEqual(game['raw_terminal_scores'], game['status']['final_scores'])
        self.assertEqual(report['summary']['completed'], 4)

    def test_both_schemas_and_initial_model(self):
        for schema in (1, 2):
            path = self.checkpoint(schema)
            report = evaluate(path, path, self.snapshot, seeds=[100], b_initial=True)
            self.assertTrue(report['models']['B']['initial'])
            self.assertEqual(report['summary']['completed'], 2)

    def test_truncation_never_awards_score_or_draw(self):
        path = self.checkpoint(max_steps=0)
        report = evaluate(path, path, self.snapshot, seeds=[100])
        self.assertEqual(report['summary']['truncated'], 2)
        self.assertEqual(report['summary']['draw'], 0)
        self.assertEqual(report['paired']['excluded_pairs'], 1)
        self.assertIsNone(report['paired']['mean_a_score_diff'])
        for game in report['games']:
            self.assertIsNone(game['raw_terminal_scores'])
            self.assertIsNone(game['result'])

    def test_invalid_seeds_and_mismatched_contract(self):
        path = self.checkpoint()
        for seeds in ([], [90], [100, 100], [True], [-1], range(257)):
            with self.assertRaises(ValueError):
                evaluate(path, path, self.snapshot, seeds=seeds)
        with self.assertRaises(ValueError):
            evaluate(path, self.checkpoint(max_steps=0), self.snapshot, seeds=[100])

    def test_hidden_changes_leave_both_seats_policy_inputs_and_probabilities_unchanged(self):
        for side, enemy_id in (('BLUE', 'red'), ('RED', 'blue')):
            episode = TwoPlayerEpisode(self.snapshot)
            with patch.object(Game, 'can_observe', return_value=False):
                episode.reset(seed=10)
                if side == 'RED':
                    episode.step(episode.legal_actions('BLUE')[0], side='BLUE')
                adapter = PublicObservationAdapter(episode)
                observation, candidates = adapter.observe(side), adapter.candidate_views(side)
                policy = NeuralPolicy(buckets=16, hidden_dim=4, representation='public_v2_split')
                before = policy.probabilities(observation, candidates)
                enemy = episode._session.env.game.units[enemy_id]
                enemy.pos = 20008
                # 不扰动 `blood`：它已通过公开的「剩余兵力分」合计反出（平台行为），
                # 见 tests/test_public_observation.py::
                #     test_public_score_reveals_only_aggregate_enemy_strength
                enemy.ammo['a5'] = 999
                enemy.detail['target'] = 'secret'
                enemy.stunned_until = 12345
                self.assertEqual(adapter.observe(side), observation)
                self.assertEqual(adapter.candidate_views(side), candidates)
                np.testing.assert_array_equal(before, policy.probabilities(adapter.observe(side), adapter.candidate_views(side)))

    def test_invalid_probabilities_rejected(self):
        policy = NeuralPolicy(buckets=16, hidden_dim=4)
        for probs in ([np.nan], [-1], [0], [0.5, 0.5]):
            with patch.object(policy, 'probabilities', return_value=probs):
                with self.assertRaises(ValueError):
                    FrozenSampler(policy, 0)({}, [{'type': 'wait'}])


if __name__ == '__main__':
    unittest.main()
