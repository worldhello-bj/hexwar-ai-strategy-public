"""Schema 4 aggregate public relations and strict legacy compatibility."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from engine.game import Game
from env.episode_api import TwoPlayerEpisode
from env.match_eval import write_synthetic_snapshot
from env.neural_policy import NeuralPolicy
from env.neural_train import NeuralTrainer
from env.public_observation import PublicObservationAdapter
from env.spatial_observation import SpatialObservationAdapter
from env.contact_observation import ContactObservationAdapter, objective_contacts
from tools.export_policy_replay import export_replay, reconstruct_observation
from tools.evaluate_frozen_opponents import run_game
from tools.cloud_train import parser, requested_config


class ContactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.snapshot = write_synthetic_snapshot(Path(self.tmp.name) / 'tokens')
        (self.snapshot / 'cities.json').write_text(json.dumps([
            {'id': 'objective', 'pos': 20006, 'name': 'Public', 'flag': None}]))

    def make(self):
        episode = TwoPlayerEpisode(self.snapshot, max_seconds=20)
        episode.reset(seed=17)
        return episode, ContactObservationAdapter(episode)

    def test_endpoint_ownership_and_shared_keys(self):
        observation = {'own': {'u': {'pos': '20004', 'typeid': '1'}},
                       'objectives': {'x': {'pos': '20006', 'held_by': None}}}
        action = {'type': 'move', 'unit': 'u', 'path': [20004, 20006]}
        expected = {'intended_endpoint_is_public_objective': True,
                    'intended_endpoint_is_unowned_public_objective': True,
                    'acting_unit_is_ground_type1or2': True}
        self.assertEqual(objective_contacts(action, observation), expected)
        observation['objectives']['another-id'] = observation['objectives'].pop('x')
        self.assertEqual(objective_contacts(action, observation), expected)
        for owner in ('BLUE', 'RED', ''):
            observation['objectives']['another-id']['held_by'] = owner
            self.assertEqual(objective_contacts(action, observation), {**expected,
                'intended_endpoint_is_unowned_public_objective': False})
        del observation['objectives']['another-id']['held_by']
        self.assertFalse(objective_contacts(action, observation)['intended_endpoint_is_unowned_public_objective'])
        action['path'] = [20004, 20006, 20008]
        self.assertFalse(objective_contacts(action, observation)['intended_endpoint_is_public_objective'])
        action['type'] = 'attack'
        observation['own']['u']['pos'] = '20006'
        self.assertTrue(objective_contacts(action, observation)['intended_endpoint_is_public_objective'])
        self.assertFalse(any(objective_contacts({'type': 'wait'}, observation).values()))
        self.assertFalse(any(objective_contacts({'type': 'move', 'unit': 'hidden', 'path': [20006]}, observation).values()))

    def test_ground_flag_does_not_claim_capture_eligibility(self):
        obs = {'own': {'u': {'typeid': '2', 'name': '火炮', 'pos': '20006',
                            'blood': 0, 'loaded_into': 'carrier'}}, 'objectives': {}}
        action = {'type': 'attack', 'unit': 'u'}
        for typeid in (1, 2, '1', '2'):
            obs['own']['u']['typeid'] = typeid
            self.assertTrue(objective_contacts(action, obs)['acting_unit_is_ground_type1or2'])
        for typeid in (3, '3', None, True):
            obs['own']['u']['typeid'] = typeid
            self.assertFalse(objective_contacts(action, obs)['acting_unit_is_ground_type1or2'])

    def test_hidden_perturbation_and_no_extra_engine_evaluation(self):
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
            with patch.object(Game, 'capture_eligible', side_effect=AssertionError('engine read')):
                for view in views:
                    self.assertEqual(adapter._candidate_context(view['action'], obs), view['public_context'])
            views[0]['public_context']['objective_contacts'].clear()
            self.assertNotEqual(adapter.candidate_views(), views)

    def test_existing_schemas_byte_identical_features_parameters_rng(self):
        path = Path(self.tmp.name) / 'baseline.py'
        path.write_text(subprocess.check_output(['git', 'show', 'e1f85ee:env/neural_policy.py'], text=True))
        spec = importlib.util.spec_from_file_location('contact_baseline', path)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        episode, _ = self.make()
        for schema, representation in ((1, 'legacy_json'), (2, 'public_v2_split'), (3, 'public_v3_spatial')):
            adapter = {1: episode, 2: PublicObservationAdapter(episode), 3: SpatialObservationAdapter(episode)}[schema]
            observation = adapter.observe('BLUE')
            candidates = adapter.legal_actions() if schema == 1 else adapter.candidate_views()
            config = dict(seed=4, buckets=32, hidden_dim=3, representation=representation)
            old, new = baseline.NeuralPolicy(**config), NeuralPolicy(**config)
            self.assertEqual(json.dumps(old.checkpoint(), sort_keys=True), json.dumps(new.checkpoint(), sort_keys=True))
            for _ in range(3):
                ca, a, pa = old.sample(observation, candidates, return_probabilities=True)
                cb, b, pb = new.sample(observation, candidates, return_probabilities=True)
                self.assertEqual(ca, cb)
                np.testing.assert_array_equal(pa, pb)
                for fa, fb in zip((a.observation, *a.candidates), (b.observation, *b.candidates)):
                    for xa, xb in zip(fa, fb):
                        self.assertEqual(xa.tobytes(), xb.tobytes())
            self.assertEqual(old.checkpoint(), new.checkpoint())

    def test_schema3_relations_and_raw_rewards_preserved(self):
        episode, adapter = self.make()
        spatial = SpatialObservationAdapter(episode)
        obs4 = adapter.observe('BLUE')
        obs4['schema_version'] = 3
        self.assertEqual(obs4, spatial.observe('BLUE'))
        for v4, v3 in zip(adapter.candidate_views(), spatial.candidate_views()):
            v4['public_context'].pop('objective_contacts')
            self.assertEqual(v4, v3)
        control = TwoPlayerEpisode(self.snapshot, max_seconds=20)
        control.reset(seed=17)
        while not adapter.status()['terminated']:
            before = episode.status()
            candidates = adapter.candidate_views()
            self.assertEqual(episode.status(), before)
            action = next(v['action'] for v in candidates if v['action']['type'] == 'wait')
            self.assertEqual(adapter.step(action), control.step(copy.deepcopy(action)))
        self.assertEqual(adapter.status()['final_scores'], control.status()['final_scores'])

    def test_runner_opt_in_and_default_unchanged(self):
        required = ["--snapshot", str(self.snapshot), "--output", self.tmp.name, "--episodes", "4"]
        self.assertEqual(requested_config(parser().parse_args(required))["observation_schema"], 2)
        selected = requested_config(parser().parse_args(required + ["--observation-schema", "4",
                            "--baseline-offset", "public_own_score"]))
        self.assertEqual(selected["observation_schema"], 4)
        self.assertEqual(selected["policy_config"]["baseline_offset"], "public_own_score")

    def test_roundtrip_replay_eval_and_mismatch_rejection(self):
        trainer = NeuralTrainer(self.snapshot, observation_schema=4,
                    episode_config={'max_seconds': 20},
                    policy_config={'buckets': 32, 'hidden_dim': 3})
        path = Path(self.tmp.name) / 'checkpoint.json'
        trainer.save(path)
        before = path.read_bytes()
        self.assertEqual(trainer.checkpoint(), NeuralTrainer.load(path, self.snapshot).checkpoint())
        report = export_replay(path, self.snapshot, seed=42, learner_side='BLUE')
        self.assertEqual(report['observation_schema'], 4)
        for decision in report['decisions']:
            obs = reconstruct_observation(report, decision['observation'])
            views = [{k: c[k] for k in ('action', 'public_context')} for c in decision['candidates']]
            np.testing.assert_array_equal(trainer.policy.probabilities(obs, views), [c['probability'] for c in decision['candidates']])
        self.assertEqual(path.read_bytes(), before)
        other = NeuralTrainer(self.snapshot, observation_schema=3, episode_config={'max_seconds': 20},
                              policy_config={'buckets': 32, 'hidden_dim': 3})
        self.assertTrue(run_game(self.snapshot, {'A': trainer.policy, 'B': other.policy},
                    {'A': 4, 'B': 3}, trainer.episode_config, seed=42, a_side='BLUE')['status']['terminated'])
        episode, adapter = self.make()
        obs, views = adapter.observe('BLUE'), adapter.candidate_views()
        for wrong_obs, wrong_views in ((SpatialObservationAdapter(episode).observe('BLUE'), views),
                                      (obs, SpatialObservationAdapter(episode).candidate_views())):
            with self.assertRaises(ValueError):
                trainer.policy.probabilities(wrong_obs, wrong_views)
        views[0]['public_context']['objective_contacts']['acting_unit_is_ground_type1or2'] = 1
        with self.assertRaises(ValueError):
            trainer.policy.probabilities(obs, views)
        data = json.loads(before)
        for section in ('policy', 'initial_policy'):
            changed = copy.deepcopy(data)
            changed[section]['config']['representation'] = 'public_v3_spatial'
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                NeuralTrainer.load(path, self.snapshot)
        data['observation_schema'] = 3
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            NeuralTrainer.load(path, self.snapshot)


if __name__ == '__main__':
    unittest.main()
