"""Learned replay provenance, fidelity, immutability and seat-only visibility."""
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from env.episode_api import TwoPlayerEpisode
from env.match_eval import write_synthetic_snapshot
from env.neural_train import NeuralTrainer
from tools.export_policy_replay import export_replay, write_csv, main, reconstruct_observation


class PolicyReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.snapshot = write_synthetic_snapshot(self.root / 'snapshot')

    def trainer(self, schema=2, public=True, seconds=40, steps=20):
        trainer = NeuralTrainer(self.snapshot, seed=13, observation_schema=schema,
            train_seed_start=10, heldout_seeds=[100], batch_episodes=2,
            policy_config={'buckets': 16, 'hidden_dim': 4},
            episode_config={'max_seconds': seconds, 'max_steps': steps, 'public_score': public})
        trainer.train(2)
        path = self.root / f'checkpoint-{schema}-{public}-{seconds}-{steps}.json'
        trainer.save(path)
        return trainer, path

    def test_determinism_original_actions_and_same_evaluation_trajectory(self):
        for schema in (1, 2):
            trainer, path = self.trainer(schema)
            for side in ('BLUE', 'RED'):
                captured = []
                original_step = TwoPlayerEpisode.step
                def record_step(episode, action, **kwargs):
                    captured.append((kwargs['side'], copy.deepcopy(action)))
                    return original_step(episode, action, **kwargs)
                with patch.object(TwoPlayerEpisode, 'step', record_step):
                    report = export_replay(path, self.snapshot, seed=100, learner_side=side)
                exported_steps = copy.deepcopy(captured)
                self.assertEqual(report, export_replay(path, self.snapshot, seed=100, learner_side=side))
                self.assertEqual([a for s, a in captured if s == side],
                                 [d['selected_action'] for d in report['decisions']])
                for decision in report['decisions']:
                    restored = reconstruct_observation(report, decision['observation'])
                    candidates = ([{'action': c['action'], 'public_context': c['public_context']}
                                   for c in decision['candidates']] if schema == 2
                                  else [c['action'] for c in decision['candidates']])
                    self.assertEqual(list(trainer.policy.probabilities(restored, candidates)),
                                     [c['probability'] for c in decision['candidates']])
                    if schema == 2:
                        self.assertNotIn('board', decision['observation'])
                        self.assertEqual(decision['observation']['board_ref'], '#/public_board')
                    selected = decision['candidates'][decision['selected_index']]
                    self.assertEqual(selected['action'], decision['selected_action'])
                    self.assertEqual(selected['probability'], decision['selected_probability'])
                    self.assertAlmostEqual(sum(c['probability'] for c in decision['candidates']), 1)
                captured.clear()
                with patch.object(TwoPlayerEpisode, 'step', record_step):
                    expected, _ = trainer._rollout(trainer.policy, 100, side, training=False)
                self.assertEqual(captured, exported_steps)
                self.assertEqual(expected['own_score'], report['ending']['own_original_score'])
                self.assertEqual(expected['status']['seconds'], report['ending']['game_time'])

    def test_entire_checkpoint_parameters_rng_and_source_are_frozen(self):
        trainer, path = self.trainer()
        clone = NeuralTrainer.load(path, self.snapshot)
        before = copy.deepcopy(clone.checkpoint())
        source = path.read_bytes()
        with patch('tools.export_policy_replay.NeuralTrainer.load', return_value=clone), \
             patch.object(clone.policy, 'update_batch', side_effect=AssertionError('must not train')), \
             patch.object(clone.policy, 'sample', side_effect=AssertionError('must not mutate sampling RNG')):
            report = export_replay(path, self.snapshot, seed=100, learner_side='BLUE')
        self.assertEqual(before, clone.checkpoint())
        self.assertEqual(source, path.read_bytes())
        self.assertTrue(report['models_and_source_checkpoint_unchanged'])

    def test_hidden_enemy_perturbation_does_not_enter_replay(self):
        units_path = self.snapshot / 'units.json'
        units = json.loads(units_path.read_text())
        units[1]['pos'] = 900090
        units[1]['name'] = 'SECRET-HIDDEN-NAME'
        units_path.write_text(json.dumps(units))
        _, path = self.trainer(seconds=10)
        reference = export_replay(path, self.snapshot, seed=100, learner_side='BLUE')
        original_reset = TwoPlayerEpisode.reset
        def perturb_reset(episode, **kwargs):
            status = original_reset(episode, **kwargs)
            enemy = episode._session.env.game.units['red']
            enemy.pos = 950095
            enemy.name = 'DIFFERENT-PRIVATE-NAME'
            return status
        with patch.object(TwoPlayerEpisode, 'reset', perturb_reset):
            changed = export_replay(path, self.snapshot, seed=100, learner_side='BLUE')
        self.assertEqual(reference, changed)
        self.assertFalse(reference['initial_observation']['enemy_visible'])
        self.assertFalse(reference['initial_observation']['enemy_known'])
        self.assertNotIn('SECRET-HIDDEN-NAME', json.dumps(reference))
        self.assertNotIn('900090', json.dumps(reference))
        self.assertNotIn('950095', json.dumps(changed))

    def test_private_score_mode_exports_only_own_terminal_payout(self):
        for schema in (1, 2):
            _, path = self.trainer(schema, public=False)
            report = export_replay(path, self.snapshot, seed=100, learner_side='BLUE')
            for obs in [report['initial_observation'], report['final_observation'],
                        *[d['observation'] for d in report['decisions']]]:
                self.assertNotIn('score', obs)
            self.assertNotIn('public_score', report['ending'])
            self.assertNotIn('final_scores', json.dumps(report))
            self.assertNotIn('winner', json.dumps(report))
            self.assertEqual([p['kind'] for p in report['review_points']], ['game_end'])
            self.assertIsNotNone(report['ending']['own_original_score'])

    def test_score_change_markers_use_public_observations_without_causes(self):
        _, path = self.trainer()
        original_step = TwoPlayerEpisode.step
        def change_score(episode, action, **kwargs):
            # Public scoring fixture: force an observable aggregate change.
            episode._session.env.game.units['blue'].score += 1
            return original_step(episode, action, **kwargs)
        with patch.object(TwoPlayerEpisode, 'step', change_score):
            report = export_replay(path, self.snapshot, seed=100, learner_side='BLUE')
        changes = [p for p in report['review_points'] if p['kind'] == 'public_score_change']
        self.assertTrue(changes)
        for point in changes:
            self.assertNotEqual(point['score'], point['previous_score'])
            self.assertNotIn('cause', point)
        self.assertEqual(report['review_points'][-1]['kind'], 'game_end')
        self.assertEqual(report['ending']['public_score'], report['final_observation']['score'])

    def test_truncation_has_no_final_score_and_csv_preserves_candidate_actions(self):
        _, path = self.trainer(seconds=100, steps=1)
        report = export_replay(path, self.snapshot, seed=100, learner_side='BLUE')
        self.assertTrue(report['ending']['truncated'])
        self.assertIsNone(report['ending']['own_original_score'])
        self.assertEqual(report['review_points'][-1]['kind'], 'truncation')
        stream = io.StringIO()
        write_csv(report, stream)
        rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
        self.assertEqual(len(rows), sum(len(d['candidates']) for d in report['decisions']))
        selected = [row for row in rows if row['selected'] == 'True']
        self.assertEqual([json.loads(row['original_action']) for row in selected],
                         [d['selected_action'] for d in report['decisions']])

    def test_cli_new_files_only_and_invalid_input(self):
        _, path = self.trainer()
        output, table = self.root / 'replay.json', self.root / 'replay.csv'
        args = ['--checkpoint', str(path), '--snapshot', str(self.snapshot), '--seed', '100',
                '--learner-side', 'BLUE', '--report', str(output), '--csv', str(table),
                '--markdown', str(self.root / 'replay.md')]
        self.assertEqual(main(args), 0)
        self.assertIn('Observed learned choices', (self.root / 'replay.md').read_text())
        self.assertIn('not a complete policy', (self.root / 'replay.md').read_text())
        before = output.read_bytes()
        with self.assertRaises(SystemExit):
            main(args)
        self.assertEqual(before, output.read_bytes())
        for seed, side in ((True, 'BLUE'), (-1, 'BLUE'), (100, 'UNKNOWN')):
            with self.assertRaises(ValueError):
                export_replay(path, self.snapshot, seed=seed, learner_side=side)


if __name__ == '__main__':
    unittest.main()
