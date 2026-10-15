"""Seeded integration invariants on the repository's original synthetic demo."""
import copy
import json
from pathlib import Path
import random
import unittest

from env.episode_api import TwoPlayerEpisode
from env.public_observation import PublicObservationAdapter


DEMO = Path(__file__).resolve().parents[1] / 'samples' / 'demo'


class RolloutInvariantTests(unittest.TestCase):
    def test_seeded_legal_rollouts_preserve_public_contract(self):
        for seed in (0, 7, 19):
            with self.subTest(seed=seed):
                episode = TwoPlayerEpisode(DEMO, max_seconds=180, max_steps=40)
                adapter = PublicObservationAdapter(episode)
                adapter.reset(seed=seed)
                chooser = random.Random(seed + 1000)
                while adapter.current_actor is not None:
                    before = adapter.status()
                    histories = copy.deepcopy(episode._session.histories)
                    rng = episode._session.env.game.rng.getstate()
                    for side in ('BLUE', 'RED'):
                        observation = adapter.observe(side)
                        json.dumps(observation, allow_nan=False)
                        self.assertEqual(observation, adapter.observe(side))
                        self.assertEqual(observation['time'], before['seconds'])
                        self.assertEqual(observation['time_left'], 180 - before['seconds'])
                        # Mutating nested output must not change the next read.
                        expected = copy.deepcopy(observation)
                        observation['board']['cells'].clear()
                        observation['own'].clear()
                        observation['enemy_known'].clear()
                        self.assertEqual(adapter.observe(side), expected)
                    candidates = adapter.candidate_views()
                    actions = adapter.legal_actions()
                    self.assertTrue(actions)
                    self.assertEqual([c['action'] for c in candidates], actions)
                    self.assertEqual(actions, adapter.legal_actions())
                    self.assertEqual(episode._session.histories, histories)
                    self.assertEqual(episode._session.env.game.rng.getstate(), rng)
                    action = chooser.choice(actions)
                    result = adapter.step(action, side=before['current_actor'])
                    json.dumps(result, allow_nan=False)
                    self.assertEqual(result['steps'], before['steps'] + 1)
                    self.assertGreaterEqual(result['seconds'], before['seconds'])
                    self.assertLessEqual(result['seconds'], 180)
                    if not result['terminated']:
                        self.assertEqual(result['rewards'], {'BLUE': 0, 'RED': 0})
                        self.assertIsNone(result['final_scores'])
                    else:
                        self.assertEqual(result['rewards'], result['final_scores'])
                    for unit in episode._session.env.game.units.values():
                        self.assertGreaterEqual(unit.blood, 0)
                        self.assertTrue(all(value >= 0 for value in unit.ammo.values()))
                    # The submitted action also remains caller-owned.
                    action.clear()
                    self.assertEqual(adapter.status()['steps'], result['steps'])
                self.assertEqual(adapter.candidate_views(), [])
                self.assertEqual(adapter.status()['rewards'], {'BLUE': 0, 'RED': 0})
                for side in ('BLUE', 'RED'):
                    self.assertTrue(adapter.observe(side)['done'])
                    self.assertEqual(adapter.legal_actions(side), [])


if __name__ == '__main__':
    unittest.main()
