"""Schema 3 geometry, public boundary, compatibility and reward invariants."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from engine import geometry as G
from engine.game import Game
from env.episode_api import TwoPlayerEpisode
from env.match_eval import write_synthetic_snapshot
from env.neural_policy import NeuralPolicy
from env.neural_train import NeuralTrainer
from env.public_observation import PublicObservationAdapter
from env.spatial_observation import SpatialObservationAdapter, objective_relations
from tools.export_policy_replay import export_replay, reconstruct_observation
from tools.evaluate_frozen_opponents import run_game


class SpatialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.snapshot = write_synthetic_snapshot(Path(self.tmp.name) / 'tokens')
        (self.snapshot / 'cities.json').write_text(json.dumps([
            {'id': 'objective', 'pos': 20006, 'name': 'Public', 'flag': None}]))

    def make(self):
        episode = TwoPlayerEpisode(self.snapshot, max_seconds=20)
        episode.reset(seed=17)
        return episode, SpatialObservationAdapter(episode)

    def test_geometry_even_odd_rows_and_signed_changes(self):
        for row in (4, 5):
            origin = G.make_mapid(row, 4)
            for dest in G.neighbors(origin):
                obs = {'own': {'u': {'pos': str(origin)}},
                       'objectives': {'o': {'pos': str(dest), 'held_by': 'RED'}}}
                action = {'type': 'move', 'unit': 'u', 'path': [origin, dest]}
                self.assertEqual(objective_relations(action, obs)['o'], {
                    'origin_distance': 1, 'intended_destination_distance': 0,
                    'distance_change': -1, 'held_by': 'RED'})
                obs['objectives']['o']['pos'] = str(origin)
                self.assertEqual(objective_relations(action, obs)['o']['distance_change'], 1)
                action['type'] = 'attack'
                self.assertEqual(objective_relations(action, obs)['o']['distance_change'], 0)
        self.assertEqual(objective_relations({'type': 'wait'}, obs), {})

    def test_schema2_inputs_actions_and_features_unchanged(self):
        episode, spatial = self.make()
        old = PublicObservationAdapter(episode)
        v2, v3 = old.observe('BLUE'), spatial.observe('BLUE')
        v3['schema_version'] = 2
        self.assertEqual(v2, v3)
        views2, views3 = old.candidate_views(), spatial.candidate_views()
        for left, right in zip(views2, views3):
            right = copy.deepcopy(right)
            right['public_context'].pop('objective_relations')
            self.assertEqual(left, right)
        # Load baseline source from the isolated clone's original commit, without
        # touching or importing the actively training baseline directory.
        source = subprocess.check_output(['git', 'show', '38fc987:env/neural_policy.py'], text=True)
        module_path = Path(self.tmp.name) / 'baseline_policy.py'
        module_path.write_text(source)
        spec = importlib.util.spec_from_file_location('baseline_policy', module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = dict(seed=4, buckets=32, hidden_dim=3, representation='public_v2_split')
        baseline, current = module.NeuralPolicy(**config), NeuralPolicy(**config)
        self.assertEqual(baseline.checkpoint(), current.checkpoint())
        np.testing.assert_array_equal(baseline.probabilities(v2, views2), current.probabilities(v2, views2))
        a, b = baseline._decision(v2, views2), current._decision(v2, views2)
        for left, right in zip((a.observation, *a.candidates), (b.observation, *b.candidates)):
            for x, y in zip(left, right):
                np.testing.assert_array_equal(x, y)

    def test_hidden_state_perturbation_and_detachment(self):
        with patch.object(Game, 'can_observe', return_value=False):
            episode, adapter = self.make()
            obs, views = adapter.observe('BLUE'), adapter.candidate_views()
            enemy = episode._session.env.game.units['red']
            enemy.pos = 20008
            enemy.ammo['a5'] = 999
            enemy.detail.update(target='secret', carrier='secret')
            enemy.stunned_until = 12345
            self.assertEqual(adapter.observe('BLUE'), obs)
            self.assertEqual(adapter.candidate_views(), views)
            next(v for v in views if v['public_context']['objective_relations'])['public_context']['objective_relations'].clear()
            self.assertNotEqual(adapter.candidate_views(), views)

    def test_reward_action_execution_and_score_unchanged(self):
        episode, adapter = self.make()
        control = TwoPlayerEpisode(self.snapshot, max_seconds=20)
        control.reset(seed=17)
        while not adapter.status()['terminated']:
            before = episode.status()
            adapter.observe(adapter.current_actor)
            candidates = adapter.candidate_views()
            self.assertEqual(episode.status(), before)
            action = next(v['action'] for v in candidates if v['action']['type'] == 'wait')
            self.assertEqual(adapter.step(action), control.step(copy.deepcopy(action)))
        self.assertEqual(adapter.status()['final_scores'], control.status()['final_scores'])

    def test_frozen_replay_and_mixed_schema_evaluation(self):
        trainers = {s: NeuralTrainer(self.snapshot, observation_schema=s,
                    episode_config={'max_seconds': 20},
                    policy_config={'buckets': 32, 'hidden_dim': 3}) for s in (2, 3)}
        trainer = trainers[3]
        checkpoint = Path(self.tmp.name) / 'replay.json'
        trainer.save(checkpoint)
        before = checkpoint.read_bytes()
        report = export_replay(checkpoint, self.snapshot, seed=42, learner_side='BLUE')
        self.assertEqual(report['observation_schema'], 3)
        for decision in report['decisions']:
            obs = reconstruct_observation(report, decision['observation'])
            candidates = [{k: c[k] for k in ('action', 'public_context')}
                          for c in decision['candidates']]
            np.testing.assert_array_equal(trainer.policy.probabilities(obs, candidates),
                         [c['probability'] for c in decision['candidates']])
        self.assertEqual(checkpoint.read_bytes(), before)
        game = run_game(self.snapshot, {'A': trainer.policy, 'B': trainers[2].policy},
                        {'A': 3, 'B': 2}, trainer.episode_config, seed=42, a_side='BLUE')
        self.assertTrue(game['status']['terminated'])

    def test_checkpoint_roundtrip_and_schema_mismatch_rejected(self):
        episode, adapter = self.make()
        obs, views = adapter.observe('BLUE'), adapter.candidate_views()
        policy = NeuralPolicy(buckets=32, hidden_dim=3, representation='public_v3_spatial')
        restored = NeuralPolicy.from_checkpoint(policy.checkpoint())
        np.testing.assert_array_equal(policy.probabilities(obs, views), restored.probabilities(obs, views))
        with self.assertRaises(ValueError):
            policy.probabilities(PublicObservationAdapter(episode).observe('BLUE'), views)
        with self.assertRaises(ValueError):
            NeuralPolicy(representation='public_v2_split').probabilities(obs, views)
        with self.assertRaises(ValueError):
            policy.probabilities(obs, PublicObservationAdapter(episode).candidate_views())
        trainer = NeuralTrainer(self.snapshot, observation_schema=3,
                                policy_config={'buckets': 32, 'hidden_dim': 3})
        path = Path(self.tmp.name) / 'checkpoint.json'
        trainer.save(path)
        restored_trainer = NeuralTrainer.load(path, self.snapshot)
        self.assertEqual(trainer.checkpoint(), restored_trainer.checkpoint())
        data = json.loads(path.read_text())
        data['observation_schema'] = 2
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            NeuralTrainer.load(path, self.snapshot)


if __name__ == '__main__':
    unittest.main()
