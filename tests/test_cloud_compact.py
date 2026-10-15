"""Cloud NPZ transport uses real trainer state, without running engine games."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from env.match_eval import write_synthetic_snapshot
from env.neural_policy import NeuralPolicy, Decision
from env.neural_train import NeuralTrainer, summary
from tools import cloud_train


def synthetic_rollout(trainer, policy, seed, side, *, training):
    observation = {'schema_version': 2, 'board': {}, 'seed': seed,
                   'score': {'own': seed + .25}}
    candidates = [{'action': i, 'public_context': {'objective_relations': []}}
                  for i in (1, 2)]
    trajectory = []
    if training:
        _, decision = policy.sample(observation, candidates)
        trajectory.append(decision)
    else:
        policy.probabilities(observation, candidates)
    score = float(seed + 1)
    return {'seed': seed, 'learner_side': side, 'own_score': score,
            'score_diff': 0., 'result': 'draw',
            'status': {'terminated': True, 'truncated': False,
                       'scoring_version': 8, 'scoring_mode': 'verified',
                       'final_scores': {'BLUE': score, 'RED': score}},
            'elapsed_seconds': 0.}, trajectory


class CloudCompactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.snapshot = write_synthetic_snapshot(self.root / 'snapshot')
        self.config = dict(seed=17, train_seed_start=10, batch_episodes=2,
            heldout_seeds=[100], observation_schema=2, allow_real=True,
            scoring_mode='verified',
            episode_config={'max_seconds': 20, 'max_steps': 100, 'move_radius': 1},
            policy_config={'buckets': 8, 'hidden_dim': 2, 'learning_rate': .001,
                           'baseline_offset': 'prior_batch_mean', 'update_mode': 'ppo_clip'})
        for patcher in (patch.object(NeuralTrainer, '_rollout', synthetic_rollout),
                        patch.object(cloud_train, 'requested_config', return_value=self.config),
                        patch.object(cloud_train, 'git_commit', return_value='test-commit')):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.replays = []

    def args(self, name='run', format='npz', episodes=4):
        return cloud_train.parser().parse_args([
            '--snapshot', str(self.snapshot), '--output', str(self.root / 'output'),
            '--run-id', name, '--checkpoint-format', format, '--episodes', str(episodes),
            '--checkpoint-every', '1', '--seed', '17', '--train-seed-start', '10',
            '--batch-episodes', '2', '--eval-seeds', '100'])

    def replay(self, checkpoint, snapshot, **kwargs):
        loaded = NeuralTrainer.load(checkpoint, snapshot, allow_real=True)
        self.replays.append((Path(checkpoint), loaded.episodes_attempted))
        return {'seed': kwargs['seed'], 'learner_side': kwargs['learner_side']}

    def run_cloud(self, args):
        return cloud_train.run(args, replay_exporter=self.replay)

    def directory(self, args):
        return args.output / args.run_id

    def checkpoint(self, args):
        path = self.directory(args) / ('checkpoint.' + args.checkpoint_format)
        return NeuralTrainer.load(path, self.snapshot, allow_real=True).checkpoint()

    def test_default_json_manifest_and_npz_explicit_manifest(self):
        self.assertEqual(cloud_train.parser().parse_args([]).checkpoint_format, 'json')
        for format in ('json', 'npz'):
            with self.subTest(format=format):
                args = self.args(format, format)
                self.assertEqual(self.run_cloud(args), 0)
                manifest = json.loads((self.directory(args) / 'manifest.json').read_text())
                if format == 'json':
                    self.assertNotIn('checkpoint_format', manifest)
                else:
                    self.assertEqual(manifest['checkpoint_format'], 'npz')
                self.assertTrue((self.directory(args) / ('checkpoint.' + format)).is_file())
                self.assertFalse((self.directory(args) / ('checkpoint.' + ('npz' if format == 'json' else 'json'))).exists())
                self.assertEqual(self.replays[-1][0].suffix, '.' + format)

    def test_resume_requires_format_source_and_config_and_preserves_files(self):
        for format in ('json', 'npz'):
            args = self.args(format, format)
            self.run_cloud(args)
            original = {p.name: p.read_bytes() for p in self.directory(args).iterdir() if p.is_file()}
            args.resume = True
            args.checkpoint_format = 'json' if format == 'npz' else 'npz'
            with self.assertRaises(ValueError):
                self.run_cloud(args)
            args.checkpoint_format = format
            for target, changed in [('source_manifest', {'sha256': 'changed'}),
                                    ('requested_config', {**self.config, 'seed': 18})]:
                with patch.object(cloud_train, target, return_value=changed), self.assertRaises(ValueError):
                    self.run_cloud(args)
            self.assertEqual(original, {p.name: p.read_bytes() for p in self.directory(args).iterdir() if p.is_file()})
            args.episodes = 6
            self.assertEqual(self.run_cloud(args), 0)
            expected = NeuralTrainer(self.snapshot, **self.config)
            expected.train(6)
            self.assertEqual(self.checkpoint(args), expected.checkpoint())

    def test_new_run_cannot_overwrite_existing_run(self):
        args = self.args()
        self.run_cloud(args)
        checkpoint = self.directory(args) / 'checkpoint.npz'
        before = checkpoint.read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_cloud(args)
        self.assertEqual(checkpoint.read_bytes(), before)

    def test_stop_after_one_episode_preserves_exact_partial_batch_and_resume(self):
        args = self.args()
        stop = self.root / 'stop'
        args.stop_file = stop
        original_train = NeuralTrainer.train
        def train_and_stop(trainer, count):
            rows = original_train(trainer, count)
            stop.touch()
            return rows
        with patch.object(NeuralTrainer, 'train', train_and_stop):
            self.assertEqual(self.run_cloud(args), 130)
        expected = NeuralTrainer(self.snapshot, **self.config)
        expected.train(1)
        self.assertEqual(len(expected.pending), 1)
        self.assertEqual(self.checkpoint(args), expected.checkpoint())
        self.assertEqual(len(self.replays), 0)
        self.assertTrue(stop.exists())
        args.resume = True
        self.assertEqual(self.run_cloud(args), 130)
        self.assertEqual(self.checkpoint(args), expected.checkpoint())
        stop.unlink()
        self.assertEqual(self.run_cloud(args), 0)
        expected.train(3)
        self.assertEqual(self.checkpoint(args), expected.checkpoint())

    def test_initialize_from_each_format_into_each_format_preserves_parents(self):
        parent = NeuralTrainer(self.snapshot, **self.config)
        rows = parent.train(2)
        journal = self.root / 'parent-training.json'
        journal.write_text(json.dumps({'training': rows, 'training_summary': summary(rows),
            'episodes_attempted': 2, 'episodes_completed': 2,
            'optimizer_updates': parent.policy.steps, 'training_elapsed_seconds': 1.25}))
        journal_bytes = journal.read_bytes()
        stop = self.root / 'stop'
        stop.touch()
        for source_format in ('json', 'npz'):
            source = self.root / ('parent.' + source_format)
            parent.save(source)
            source_bytes = source.read_bytes()
            for target_format in ('json', 'npz'):
                with self.subTest(source=source_format, target=target_format):
                    args = self.args(source_format + '-' + target_format, target_format)
                    args.initialize_from, args.parent_training_report = source, journal
                    args.stop_file = stop
                    self.assertEqual(self.run_cloud(args), 130)
                    self.assertEqual(self.checkpoint(args), parent.checkpoint())
                    imported = json.loads((self.directory(args) / 'training.json').read_text())
                    self.assertEqual(imported['training'], rows)
                    self.assertEqual(imported['training_elapsed_seconds'], 1.25)
                    audit = json.loads((self.directory(args) / 'import-audit.json').read_text())
                    self.assertEqual(audit['parent_checkpoint_sha256'], cloud_train.file_sha256(source))
                    self.assertEqual(audit['parent_training_report_sha256'], cloud_train.file_sha256(journal))
                    self.assertTrue(audit['parameters_optimizer_rng_preserved'])
                    self.assertEqual(source.read_bytes(), source_bytes)
                    self.assertEqual(journal.read_bytes(), journal_bytes)
                    args.resume = True
                    args.initialize_from = args.parent_training_report = None
                    stop.unlink()
                    self.assertEqual(self.run_cloud(args), 0)
                    expected = NeuralTrainer.load(source, self.snapshot, allow_real=True)
                    expected.train(2)
                    self.assertEqual(self.checkpoint(args), expected.checkpoint())
                    self.assertEqual(source.read_bytes(), source_bytes)
                    self.assertEqual(journal.read_bytes(), journal_bytes)
                    stop.touch()

    def test_npz_save_uses_new_temporary_file_and_atomic_replacement(self):
        args = self.args()
        target = self.directory(args) / 'checkpoint.npz'
        save = NeuralTrainer.save
        replace = cloud_train.os.replace
        saved_paths, replaced = [], []
        def checked_save(trainer, path):
            path = Path(path)
            self.assertNotEqual(path, target)
            self.assertEqual(path.parent, target.parent)
            self.assertEqual(path.suffix, '.npz')
            self.assertFalse(path.exists())
            saved_paths.append(path)
            return save(trainer, path)
        def checked_replace(source, destination):
            if Path(destination) == target:
                self.assertIn(Path(source), saved_paths)
                self.assertEqual(NeuralTrainer.load(source, self.snapshot, allow_real=True).episodes_attempted,
                                 json.loads((target.parent / 'training.json').read_text())['episodes_attempted'])
                replaced.append(Path(source))
            return replace(source, destination)
        with patch.object(NeuralTrainer, 'save', checked_save), patch.object(cloud_train.os, 'replace', checked_replace):
            self.assertEqual(self.run_cloud(args), 0)
        self.assertGreaterEqual(len(replaced), 3)
        self.assertEqual(len(saved_paths), len(replaced))
        self.assertTrue(all(not p.exists() for p in saved_paths))

    def test_failed_atomic_save_preserves_checkpoint_and_cleans_owned_partial_file(self):
        args = self.args()
        self.run_cloud(args)
        target = self.directory(args) / 'checkpoint.npz'
        before = target.read_bytes()
        files_before = {p.name for p in target.parent.iterdir()}
        trainer = NeuralTrainer.load(target, self.snapshot, allow_real=True)
        partial_paths = []
        def partial_save(path):
            path = Path(path)
            self.assertFalse(path.exists())
            self.assertNotEqual(path, target)
            partial_paths.append(path)
            with path.open('xb') as stream:
                stream.write(b'partial invalid archive')
            raise OSError('simulated interrupted save')
        with patch.object(trainer, 'save', side_effect=partial_save):
            with self.assertRaisesRegex(OSError, 'simulated interrupted save'):
                cloud_train.atomic_npz(target, trainer)
        self.assertEqual(len(partial_paths), 1)
        self.assertFalse(partial_paths[0].exists())
        self.assertEqual(target.read_bytes(), before)
        self.assertEqual({p.name for p in target.parent.iterdir()}, files_before)
        self.assertEqual(NeuralTrainer.load(target, self.snapshot, allow_real=True).checkpoint(),
                         trainer.checkpoint())

    def test_evaluation_mutation_guard_detects_arrays_optimizer_initial_and_rng(self):
        mutations = {
            'params': lambda t: t.policy.params['value'].__setitem__(0, t.policy.params['value'][0] + 1),
            'optimizer': lambda t: t.policy.m['value'].__setitem__(0, t.policy.m['value'][0] + 1),
            'initial': lambda t: t.initial_policy['params']['value'].__setitem__(0, t.initial_policy['params']['value'][0] + 1),
            'rng': lambda t: t.policy.rng.random(),
        }
        evaluate = NeuralTrainer.evaluate
        for name, mutation in mutations.items():
            args = self.args(name)
            def corrupt(trainer):
                result = evaluate(trainer)
                mutation(trainer)
                return result
            with self.subTest(mutation=name), patch.object(NeuralTrainer, 'evaluate', corrupt):
                with self.assertRaisesRegex(RuntimeError, 'evaluation.*(changed|mutat)'):
                    self.run_cloud(args)
            expected = NeuralTrainer(self.snapshot, **self.config)
            expected.train(args.episodes)
            self.assertEqual(self.checkpoint(args), expected.checkpoint())
            state = json.loads((self.directory(args) / 'progress.json').read_text())
            self.assertEqual(state['status'], 'failed')
            self.assertFalse((self.directory(args) / 'evaluation.json').exists())
            self.assertFalse((self.directory(args) / 'replay.json').exists())

    def test_npz_save_and_real_evaluation_never_materialize_numeric_lists(self):
        trainer_checkpoint = NeuralTrainer.checkpoint
        policy_checkpoint = NeuralPolicy.checkpoint
        decision_dict = Decision.to_dict
        def trainer_arrays(trainer, *, arrays=False):
            self.assertTrue(arrays, 'trainer checkpoint list path used')
            return trainer_checkpoint(trainer, arrays=arrays)
        def policy_arrays(policy, *, arrays=False):
            self.assertTrue(arrays, 'policy checkpoint list path used')
            return policy_checkpoint(policy, arrays=arrays)
        def decision_arrays(decision, *, arrays=False):
            self.assertTrue(arrays, 'pending decision list path used')
            return decision_dict(decision, arrays=arrays)
        with patch.object(NeuralTrainer, 'checkpoint', trainer_arrays), \
             patch.object(NeuralPolicy, 'checkpoint', policy_arrays), \
             patch.object(Decision, 'to_dict', decision_arrays):
            self.assertEqual(self.run_cloud(self.args(episodes=3)), 0)


if __name__ == '__main__':
    unittest.main()
