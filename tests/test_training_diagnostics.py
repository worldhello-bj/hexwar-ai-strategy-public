import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from env.neural_policy import NeuralPolicy
from env.neural_train import NeuralTrainer
from env.match_eval import write_synthetic_snapshot
from env.training_diagnostics import EpochDiagnostics
from tools.cloud_train import parser
from tests.test_sparse_backend import HAS_SCIPY, SCIPY_REASON


class DiagnosticsTests(unittest.TestCase):
    def test_scalar_definitions_seats_and_clipping(self):
        diag = EpochDiagnostics(['BLUE', 'RED', 'BLUE'], .2)
        probabilities = np.array([.25, .75])
        for i, (adv, ratio, clipped) in enumerate([(2, 1.3, True), (-3, .7, True), (0, 1.4, False)]):
            diag.observe(i, adv, probabilities, np.log(probabilities), math.log(ratio), clipped)
        report = diag.report()
        all_stats = report['ALL']
        self.assertEqual(all_stats['advantage']['count'], 3)
        self.assertEqual(all_stats['advantage']['nonzero_count'], 2)
        self.assertEqual(all_stats['advantage']['positive_count'], 1)
        self.assertEqual(all_stats['advantage']['negative_count'], 1)
        self.assertAlmostEqual(all_stats['advantage']['mean'], -1/3)
        self.assertEqual(all_stats['sampled_ratio_clip_fraction'], 1)
        self.assertAlmostEqual(all_stats['surrogate_clip_fraction'], 2/3)
        self.assertAlmostEqual(all_stats['sampled_approx_kl']['mean'], -sum(map(math.log, [1.3, .7, 1.4])) / 3)
        self.assertEqual(report['BLUE']['advantage']['count'], 2)
        self.assertEqual(report['RED']['advantage']['count'], 1)
        self.assertIsNone(report['UNKNOWN']['sampled_ratio_clip_fraction'])

    def test_exact_policy_state_and_forward_count_on_off(self):
        for mode in ('reinforce', 'ppo_clip'):
            for backend in (('numpy', 'scipy_csr') if HAS_SCIPY else ('numpy',)):
                for baseline in ('none', 'prior_batch_mean', 'public_own_score'):
                    a = NeuralPolicy(seed=42, buckets=16, hidden_dim=3, update_mode=mode,
                                     projection_backend=backend, baseline_offset=baseline)
                    b = NeuralPolicy.from_checkpoint(a.checkpoint())
                    obs = {'score': {'own': 4}, 'time': 0}
                    candidates = [{'type': 'wait'}, {'type': 'move'}]
                    decisions = [a._decision(obs, candidates, i) for i in (0, 1)]
                    batch = [(decisions, 8), ([decisions[0]], 2), ([], 5)]
                    with patch.object(a, '_forward', wraps=a._forward) as fa, patch.object(b, '_forward', wraps=b._forward) as fb:
                        ra = a.update_batch(batch)
                        rb = b.update_batch(batch, diagnostic_seats=['BLUE', 'RED', 'BLUE'])
                        self.assertEqual(fa.call_count, fb.call_count)
                    diagnostics = rb.pop('diagnostics')
                    self.assertEqual(ra, rb)
                    self.assertEqual(a.checkpoint(), b.checkpoint())
                    first = diagnostics['epochs'][0]
                    self.assertEqual(first['ALL']['advantage']['count'], 3)
                    self.assertEqual(first['BLUE']['advantage']['count'], 2)
                    self.assertEqual(first['RED']['advantage']['count'], 1)
                    if mode == 'ppo_clip':
                        self.assertEqual(first['ALL']['sampled_ratio_clip_fraction'], 0)
                        self.assertEqual(first['ALL']['sampled_approx_kl']['mean'], 0)
                        for epoch in diagnostics['epochs']:
                            self.assertEqual(epoch['ALL']['advantage'], first['ALL']['advantage'])
                    else:
                        self.assertIsNone(first['ALL']['sampled_ratio_clip_fraction'])

    def test_trainer_state_pending_resume_unknown_and_rollout_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = write_synthetic_snapshot(Path(tmp) / 'tokens')
            config = dict(batch_episodes=2, episode_config={'max_seconds': 20},
                          policy_config={'buckets': 16, 'hidden_dim': 3, 'update_mode': 'ppo_clip'})
            a, b = NeuralTrainer(snapshot, **config), NeuralTrainer(snapshot, **config)
            with patch.object(a, '_rollout', wraps=a._rollout) as fa, patch.object(b, '_rollout', wraps=b._rollout) as fb:
                a.train(1)
                b.train(1, diagnostics=True)
                self.assertEqual(a.checkpoint(), b.checkpoint())
                rows_a = a.train(1)
                rows_b = b.train(1, diagnostics=True)
                self.assertEqual(fa.call_count, fb.call_count)
            self.assertEqual(a.checkpoint(), b.checkpoint())
            report = rows_b[0]['update'].pop('diagnostics')
            self.assertEqual(rows_a[0]['update'], rows_b[0]['update'])
            self.assertGreater(report['epochs'][0]['BLUE']['advantage']['count'], 0)
            self.assertGreater(report['epochs'][0]['RED']['advantage']['count'], 0)
            b.train(1, diagnostics=True)
            path = Path(tmp) / 'partial.npz'
            b.save(path)
            restored = NeuralTrainer.load(path, snapshot)
            report = restored.train(1, diagnostics=True)[0]['update']['diagnostics']['epochs'][0]
            self.assertGreater(report['UNKNOWN']['advantage']['count'], 0)
            self.assertEqual(report['BLUE']['advantage']['count'], 0)
            self.assertGreater(report['RED']['advantage']['count'], 0)

    @unittest.skipUnless(HAS_SCIPY, SCIPY_REASON)
    def test_schema3_schema4_csr_npz_diagnostic_parity(self):
        from tools.export_policy_replay import export_replay
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = write_synthetic_snapshot(Path(tmp) / 'tokens')
            for schema in (3, 4):
                config = dict(observation_schema=schema, batch_episodes=2,
                              episode_config={'max_seconds': 20}, policy_config={
                                  'buckets': 16, 'hidden_dim': 3, 'update_mode': 'ppo_clip',
                                  'baseline_offset': 'public_own_score', 'projection_backend': 'scipy_csr'})
                a, b = NeuralTrainer(snapshot, **config), NeuralTrainer(snapshot, **config)
                a.train(2)
                rows = b.train(2, diagnostics=True)
                self.assertIn('diagnostics', rows[-1]['update'])
                self.assertEqual(a.checkpoint(), b.checkpoint())
                path = Path(tmp) / f'schema-{schema}.npz'
                b.save(path)
                restored = NeuralTrainer.load(path, snapshot)
                self.assertEqual(restored.checkpoint(), b.checkpoint())
                self.assertEqual(export_replay(path, snapshot, seed=42, learner_side='BLUE')['observation_schema'], schema)

    def test_empty_invalid_and_cli(self):
        policy = NeuralPolicy(buckets=8, hidden_dim=2)
        before = policy.checkpoint()
        report = policy.update_batch([], diagnostic_seats=[])['diagnostics']['epochs'][0]
        self.assertEqual(report['ALL']['advantage']['count'], 0)
        self.assertEqual(before, policy.checkpoint())
        with self.assertRaises(ValueError):
            policy.update_batch([([], 1)], diagnostic_seats=[])
        args = ['--snapshot', '/tmp/input', '--output', '/tmp/output', '--episodes', '2']
        self.assertFalse(parser().parse_args(args).training_diagnostics)
        self.assertTrue(parser().parse_args(args + ['--training-diagnostics']).training_diagnostics)


if __name__ == '__main__':
    unittest.main()
