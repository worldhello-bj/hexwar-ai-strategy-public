import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from tools import cloud_train


class FakeTrainer:
    def __init__(self, snapshot, **config):
        self.manifest = {'sha256': 'test-scenario'}
        self.heldout_seeds = config['heldout_seeds']
        self.episodes_attempted = self.episodes_completed = 0
        self.policy = SimpleNamespace(steps=0)
        self.config = config

    def checkpoint(self):
        return {'attempted': self.episodes_attempted, 'completed': self.episodes_completed,
                'steps': self.policy.steps, 'config': self.config}

    @classmethod
    def load(cls, path, snapshot, **kwargs):
        data = json.loads(Path(path).read_text())
        result = cls(snapshot, **data['config'])
        result.episodes_attempted, result.episodes_completed = data['attempted'], data['completed']
        result.policy.steps = data['steps']
        return result

    def train(self, count):
        assert count == 1
        self.episodes_attempted += 1
        self.episodes_completed += 1
        self.policy.steps += 1
        return [{'seed': self.episodes_attempted - 1, 'learner_side': 'BLUE',
                 'own_score': 2, 'score_diff': 0, 'result': 'draw'}]

    def evaluate(self):
        games = [{'seed': seed, 'learner_side': side, 'own_score': 2, 'score_diff': 0, 'result': 'draw'}
                 for seed in self.heldout_seeds for side in ['BLUE', 'RED']]
        return {label: {'games': copy.deepcopy(games), 'summary': cloud_train.summary(games)}
                for label in ['initial', 'trained']}


class CloudRunnerTests(unittest.TestCase):
    def arguments(self, directory, *extra):
        return cloud_train.parser().parse_args(['--output', directory, '--run-id', 'test',
                                                '--episodes', '3', *extra])

    def replay(self, checkpoint, snapshot, **kwargs):
        self.assertTrue(kwargs['allow_real'])
        return {'seed': kwargs['seed'], 'learner_side': kwargs['learner_side']}

    def test_bounded_run_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory)
            self.assertEqual(cloud_train.run(args, FakeTrainer, self.replay), 0)
            state = json.loads((Path(directory) / 'progress.json').read_text())
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(state['episodes_attempted'], 3)
            self.assertEqual(state['episodes_completed'], 3)
            self.assertEqual(state['optimizer_updates'], 3)
            self.assertTrue(Path(state['latest_evaluation']).is_file())
            args.resume, args.episodes = True, 5
            self.assertEqual(cloud_train.run(args, FakeTrainer, self.replay), 0)
            saved = json.loads((Path(directory) / 'test/training.json').read_text())
            self.assertEqual(len(saved['training']), 5)

    def test_resume_refuses_changed_sources_or_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory)
            cloud_train.run(args, FakeTrainer, self.replay)
            args.resume = True
            with patch.object(cloud_train, 'source_manifest', return_value={'sha256': 'changed'}):
                with self.assertRaisesRegex(ValueError, 'incompatible'):
                    cloud_train.run(args, FakeTrainer, self.replay)
            args.seed += 1
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                cloud_train.run(args, FakeTrainer, self.replay)

    def test_observation_schema_default_opt_in_and_resume_guard(self):
        self.assertEqual(self.arguments('/tmp/unused').observation_schema, 2)
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory, '--observation-schema', '3')
            self.assertEqual(cloud_train.run(args, FakeTrainer, self.replay), 0)
            manifest = json.loads((Path(directory) / 'test/manifest.json').read_text())
            self.assertEqual(manifest['config']['observation_schema'], 3)
            args.resume, args.observation_schema = True, 2
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                cloud_train.run(args, FakeTrainer, self.replay)

    def test_overlap_rejected_before_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory, '--eval-seeds', '1')
            with self.assertRaisesRegex(ValueError, 'overlaps'):
                cloud_train.run(args, FakeTrainer, self.replay)
            self.assertFalse((Path(directory) / 'test').exists())

    def test_safe_stop_saves_completed_episode(self):
        import signal
        class StopTrainer(FakeTrainer):
            def train(self, count):
                rows = super().train(count)
                signal.raise_signal(signal.SIGTERM)
                return rows
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory)
            self.assertEqual(cloud_train.run(args, StopTrainer, self.replay), 130)
            state = json.loads((Path(directory) / 'progress.json').read_text())
            self.assertEqual(state['status'], 'interrupted')
            saved = json.loads((Path(directory) / 'test/checkpoint.json').read_text())
            self.assertEqual(saved['attempted'], 1)
            self.assertIsNone(state['latest_evaluation'])

    def test_stop_file_saves_episode_boundary_and_is_not_consumed(self):
        with tempfile.TemporaryDirectory() as directory:
            stop_file = Path(directory) / 'please-stop'
            class FileStopTrainer(FakeTrainer):
                def train(self, count):
                    rows = super().train(count)
                    stop_file.touch()
                    return rows
            args = self.arguments(directory, '--stop-file', str(stop_file))
            self.assertEqual(cloud_train.run(args, FileStopTrainer, self.replay), 130)
            state = json.loads((Path(directory) / 'progress.json').read_text())
            self.assertEqual(state['status'], 'interrupted')
            self.assertEqual(state['stop_reason'], 'stop_file')
            saved = json.loads((Path(directory) / 'test/checkpoint.json').read_text())
            self.assertEqual(saved['attempted'], 1)
            self.assertTrue(stop_file.exists())
            self.assertIsNone(state['latest_evaluation'])
            args.resume = True
            self.assertEqual(cloud_train.run(args, FakeTrainer, self.replay), 130)
            self.assertEqual(json.loads((Path(directory) / 'test/checkpoint.json').read_text())['attempted'], 1)

    def test_default_stop_file_blocks_training(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory)
            cloud_train.run(args, FakeTrainer, self.replay)
            (Path(directory) / 'test/STOP_REQUESTED').touch()
            args.resume, args.episodes = True, 5
            self.assertEqual(cloud_train.run(args, FakeTrainer, self.replay), 130)
            self.assertEqual(json.loads((Path(directory) / 'test/checkpoint.json').read_text())['attempted'], 3)

    def test_failure_retains_last_safe_checkpoint(self):
        class FailingTrainer(FakeTrainer):
            def train(self, count):
                self.policy.steps = 999
                raise RuntimeError('deliberate failure')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'deliberate'):
                cloud_train.run(self.arguments(directory), FailingTrainer, self.replay)
            saved = json.loads((Path(directory) / 'test/checkpoint.json').read_text())
            self.assertEqual(saved['steps'], 0)
            state = json.loads((Path(directory) / 'progress.json').read_text())
            self.assertEqual(state['status'], 'failed')
            self.assertIn('deliberate', state['issues'][0])


if __name__ == '__main__':
    unittest.main()
