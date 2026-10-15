import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from env.behavior_diagnostics import EpisodeBehaviorDiagnostics
from env.episode_api import TwoPlayerEpisode
from env.match_eval import write_synthetic_snapshot
from env.neural_train import NeuralTrainer
from tools import cloud_train
from tests.test_cloud_train import FakeTrainer
from tests.test_sparse_backend import HAS_SCIPY, SCIPY_REASON


class PublicOnly(dict):
    def __getitem__(self, key):
        if key not in ('own', 'objectives', 'enemy_visible'):
            raise AssertionError('non-public diagnostics access: ' + key)
        return super().__getitem__(key)


class BehaviorDiagnosticsTests(unittest.TestCase):
    def test_exact_counter_semantics_and_no_input_mutation(self):
        d = EpisodeBehaviorDiagnostics()
        observation = PublicOnly(own=[{'id': 1, 'pos': 0}],
                                 objectives=[{'id': 2, 'held_by': None}], enemy_visible=[],
                                 enemy_known=[{'id': 99}], private='must not read')
        original = copy.deepcopy(observation)
        action = {'type': 'move', 'unit': 1, 'path': [0, 2, 4]}
        d.observe_decision(observation, action)
        self.assertEqual(observation, original)
        self.assertEqual(action['path'], [0, 2, 4])
        # Changed positions need not match the request; do not infer execution.
        observation['own'] = {'1': {'id': '1', 'pos': '2'}}
        observation['objectives'] = {'2': {'id': '2', 'held_by': 'RED'}}
        observation['enemy_visible'] = {'99': {'id': '99'}}
        d.observe_decision(observation, {'type': 'attack', 'unit': 1})
        d.observe_decision(observation, {'type': 'move', 'unit': 1, 'path': [2]})
        r = d.report()
        self.assertEqual(r['learner_action_type_counts'], {'attack': 1, 'move': 2})
        self.assertEqual(r['unique_move_requested_units'], 1)
        self.assertEqual(r['requested_move_path_cell_count'], 4)
        self.assertEqual(r['requested_move_path_consecutive_segment_count'], 2)
        self.assertEqual(r['visible_contact_present_decisions'], 2)
        self.assertEqual(r['visible_contact_observation_count'], 3)
        self.assertEqual(r['public_objective_owner_comparisons'], 2)
        self.assertEqual(r['public_objective_owner_changes_between_learner_observations'], 1)
        self.assertEqual(r['own_position_comparisons_between_learner_observations'], 2)
        self.assertEqual(r['own_position_changes_between_learner_observations'], 1)
        self.assertEqual(r['unique_own_units_with_observed_position_change'], 1)
        self.assertEqual(r['own_hex_displacement_between_learner_observations_sum'], 1)
        json.dumps(r, allow_nan=False)

    def test_empty_and_missing_observations_have_no_invented_comparisons(self):
        d = EpisodeBehaviorDiagnostics()
        self.assertEqual(d.report()['learner_action_type_counts'], {})
        d.observe_decision({}, {'type': 'wait'})
        self.assertEqual(d.report()['visible_contact_observation_count'], 0)
        d.observe_decision({'own': [], 'objectives': []}, {'type': 'wait'})
        self.assertEqual(d.report()['public_objective_owner_comparisons'], 0)

    def test_on_off_exact_checkpoint_outcomes_inputs_and_public_call_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = write_synthetic_snapshot(Path(tmp) / 'tokens')
            for schema in (1, 2, 3, 4):
                for mode in ('reinforce', 'ppo_clip'):
                    for backend in (('numpy', 'scipy_csr') if HAS_SCIPY else ('numpy',)):
                        config = dict(observation_schema=schema, batch_episodes=2,
                                      episode_config={'max_seconds': 20}, policy_config={
                                          'buckets': 16, 'hidden_dim': 3, 'update_mode': mode,
                                          'projection_backend': backend})
                        a, b = NeuralTrainer(snapshot, **config), NeuralTrainer(snapshot, **config)
                        traces = []
                        for trainer, enabled in ((a, False), (b, True)):
                            observations, steps, samples = [], [], []
                            observe, step, sample = TwoPlayerEpisode.observe, TwoPlayerEpisode.step, trainer.policy.sample
                            def observed(episode, side):
                                value = observe(episode, side)
                                observations.append((side, copy.deepcopy(value)))
                                return value
                            def stepped(episode, action, *, side):
                                steps.append((side, copy.deepcopy(action)))
                                return step(episode, action, side=side)
                            def sampled(obs, candidates, **kwargs):
                                samples.append(copy.deepcopy((obs, candidates)))
                                return sample(obs, candidates, **kwargs)
                            with patch.object(TwoPlayerEpisode, 'observe', observed), patch.object(TwoPlayerEpisode, 'step', stepped), patch.object(trainer.policy, 'sample', sampled), patch.object(trainer.policy, '_forward', wraps=trainer.policy._forward) as forward:
                                rows = trainer.train(2, behavior_diagnostics=enabled)
                            for row in rows:
                                row.pop('elapsed_seconds')
                                if enabled:
                                    counters = row.pop('behavior_diagnostics')
                                    self.assertEqual(sum(counters['learner_action_type_counts'].values()), row['decisions'])
                                    self.assertEqual(counters['visible_contact_observation_count'], row['decisions'])
                            traces.append((rows, observations, steps, samples, forward.call_count))
                        self.assertEqual(traces[0], traces[1])
                        self.assertEqual(a.checkpoint(), b.checkpoint())

    @unittest.skipUnless(HAS_SCIPY, SCIPY_REASON)
    def test_frozen_opponent_pending_resume_and_combined_flags(self):
        from env.frozen_pool import describe_pool
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = write_synthetic_snapshot(Path(tmp) / 'tokens')
            config = dict(observation_schema=4, batch_episodes=2,
                          episode_config={'max_seconds': 20}, policy_config={
                              'buckets': 16, 'hidden_dim': 3, 'update_mode': 'ppo_clip',
                              'projection_backend': 'scipy_csr'})
            source = NeuralTrainer(snapshot, **config)
            source.train(2)  # tiny synthetic fixture required by frozen-pool validation
            source_path = Path(tmp) / 'frozen.npz'
            source.save(source_path)
            pool = describe_pool([source_path])
            source_bytes = source_path.read_bytes()
            a, b = (NeuralTrainer(snapshot, **config, opponent_pool=pool) for _ in range(2))
            frozen_before = b._opponent_pool.state_digest()
            for iteration in range(2):
                rows_a = a.train(1, diagnostics=True)
                rows_b = b.train(1, diagnostics=True, behavior_diagnostics=True)
                for row in rows_a + rows_b:
                    row.pop('elapsed_seconds')
                    row.pop('behavior_diagnostics', None)
                self.assertEqual(rows_a, rows_b)
                self.assertEqual(a.checkpoint(), b.checkpoint())
                # Resume both pending buffers with identical provenance handling.
                path_a, path_b = Path(tmp) / f'a-{iteration}.npz', Path(tmp) / f'b-{iteration}.npz'
                a.save(path_a)
                b.save(path_b)
                a = NeuralTrainer.load(path_a, snapshot)
                b = NeuralTrainer.load(path_b, snapshot)
                self.assertEqual(a.checkpoint(), b.checkpoint())
            self.assertEqual(b._opponent_pool.state_digest(), frozen_before)
            self.assertEqual(source_path.read_bytes(), source_bytes)

    def test_truncated_episode_still_reports_selected_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = write_synthetic_snapshot(Path(tmp) / 'tokens')
            trainer = NeuralTrainer(snapshot, episode_config={'max_steps': 1},
                                    policy_config={'buckets': 16, 'hidden_dim': 3})
            row = trainer.train(1, behavior_diagnostics=True)[0]
            self.assertTrue(row['status']['truncated'])
            self.assertEqual(sum(row['behavior_diagnostics']['learner_action_type_counts'].values()), row['decisions'])
            self.assertEqual(trainer.updates, 0)

    def test_flag_manifest_and_resume_guard(self):
        class DiagnosticFake(FakeTrainer):
            def train(self, count, *, behavior_diagnostics=False):
                assert behavior_diagnostics
                rows = super().train(count)
                rows[0]['behavior_diagnostics'] = {'schema_version': 1}
                return rows
        self.assertFalse(cloud_train.parser().parse_args(['--output', '/tmp/unused']).behavior_diagnostics)
        with tempfile.TemporaryDirectory() as tmp:
            args = cloud_train.parser().parse_args(['--output', tmp, '--run-id', 'test', '--episodes', '2', '--behavior-diagnostics'])
            cloud_train.run(args, DiagnosticFake, lambda *a, **k: {})
            manifest = json.loads((Path(tmp) / 'test/manifest.json').read_text())
            self.assertTrue(manifest['behavior_diagnostics_enabled'])
            args.resume = True
            args.behavior_diagnostics = False
            with self.assertRaisesRegex(ValueError, 'behavior diagnostics flag differs'):
                cloud_train.run(args, DiagnosticFake, lambda *a, **k: {})


if __name__ == '__main__':
    unittest.main()
