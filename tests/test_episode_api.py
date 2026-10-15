"""Episode contract regressions on original unarmed abstract-token fixtures."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from env.episode_api import TwoPlayerEpisode
from env.match_eval import write_synthetic_snapshot


class EpisodeApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.snapshot = write_synthetic_snapshot(Path(self.tmp.name) / 'tokens')

    def make(self, **kwargs):
        episode = TwoPlayerEpisode(self.snapshot, **kwargs)
        episode.reset(seed=17)
        return episode

    @staticmethod
    def wait(episode):
        return next(a for a in episode.legal_actions() if a['type'] == 'wait')

    def test_lifecycle_and_single_clock(self):
        episode = TwoPlayerEpisode(self.snapshot, max_seconds=25)
        for call in (lambda: episode.current_actor, episode.status,
                     lambda: episode.observe('BLUE'), episode.legal_actions,
                     lambda: episode.step({'type': 'wait', 'seconds': 10})):
            with self.assertRaises(RuntimeError):
                call()
        initial = episode.reset(seed=3)
        self.assertEqual(initial['current_actor'], 'BLUE')
        self.assertEqual(initial['rewards'], {'BLUE': 0, 'RED': 0})
        for side, seconds in [('BLUE', 10), ('RED', 20), ('BLUE', 25)]:
            self.assertEqual(episode.current_actor, side)
            state = episode.step(self.wait(episode), side=side)
            self.assertEqual(state['seconds'], seconds)
        self.assertTrue(state['terminated'])
        self.assertFalse(state['truncated'])
        self.assertEqual(state['reason'], 'game_clock')
        # 口径：剩余兵力分 = Σ上阵(score×blood)；合成快照的 token 是 score=1 / blood=4
        self.assertEqual(state['final_scores'], {'BLUE': 4, 'RED': 4})
        self.assertEqual(state['rewards'], state['final_scores'])
        self.assertIsNone(state['winner'])
        self.assertIsNone(episode.current_actor)
        self.assertEqual(episode.legal_actions(), [])
        for side in ('BLUE', 'RED'):
            self.assertTrue(episode.observe(side)['done'])
            self.assertEqual(episode.legal_actions(side), [])
        self.assertEqual(episode.status()['rewards'], {'BLUE': 0, 'RED': 0})
        with self.assertRaises(RuntimeError):
            episode.step({'type': 'wait', 'seconds': 10})

    def test_external_budget_has_no_terminal_payout(self):
        episode = self.make(max_steps=1, max_seconds=100)
        state = episode.step(self.wait(episode))
        self.assertFalse(state['terminated'])
        self.assertTrue(state['truncated'])
        self.assertEqual(state['reason'], 'step_budget')
        self.assertIsNone(state['final_scores'])
        self.assertEqual(state['rewards'], {'BLUE': 0, 'RED': 0})
        self.assertTrue(episode.observe('BLUE')['done'])
        self.assertTrue(episode.observe('RED')['done'])

    def test_endpoint_precedence_and_zero_budgets(self):
        self.assertEqual(self.make(max_steps=0).status()['reason'], 'step_budget')
        self.assertEqual(self.make(max_steps=0, max_seconds=0).status()['reason'], 'game_clock')
        episode = self.make(max_steps=1, max_seconds=10)
        self.assertEqual(episode.step(self.wait(episode))['reason'], 'game_clock')
        units_file = self.snapshot / 'units.json'
        units = json.loads(units_file.read_text())
        units_file.write_text(json.dumps(units[:1]))
        state = self.make(max_steps=0, max_seconds=0).status()
        self.assertEqual(state['reason'], 'one_side_empty')
        self.assertEqual(state['winner'], 'BLUE')
        self.assertEqual(state['final_scores'], {'BLUE': 4, 'RED': 0})
        units_file.write_text('[]')
        self.assertEqual(self.make().status()['reason'], 'both_sides_empty')

    def test_outputs_are_detached_json_and_seat_specific(self):
        episode = self.make()
        for side in ('BLUE', 'RED'):
            observation = episode.observe(side)
            self.assertEqual([u['id'] for u in observation['own']], [side.lower()])
            self.assertEqual(observation['score'], {'own': 4, 'enemy': 4, 'diff': 0})
            observation['own'][0]['blood'] = -9
            self.assertEqual(episode.observe(side)['own'][0]['blood'], 4)
            json.dumps(observation, allow_nan=False)
        self.assertEqual(episode.legal_actions('RED'), [])
        choices = episode.legal_actions()
        choices.clear()
        self.assertTrue(episode.legal_actions())
        state = episode.status()
        state['rewards']['BLUE'] = 100
        self.assertEqual(episode.status()['rewards']['BLUE'], 0)
        episode = self.make(max_seconds=0)
        state = episode.status()
        state['final_scores']['BLUE'] = 777
        self.assertEqual(episode.status()['final_scores']['BLUE'], 4)

    def test_invalid_actions_are_atomic_including_rng_and_histories(self):
        episode = self.make()
        before = episode.status()
        observations = [episode.observe(s) for s in ('BLUE', 'RED')]
        histories = copy.deepcopy(episode._session.histories)
        # Engine state serialization is not a public API; snapshot all mutable
        # attributes through deepcopy for targeted state fields and RNG below.
        rng = episode._session.env.game.rng.getstate()
        invalid = [None, [], {'type': 'nope'}, {'type': 'wait', 'seconds': -1},
                   {'type': 'wait', 'seconds': True}, {'type': 'wait', 'seconds': 10.0},
                   {'type': 'wait', 'seconds': float('nan')},
                   {'type': 'hide', 'unit': 'red', 'on': 1},
                   {'type': 'wait', 'seconds': 10, 'extra': 1}]
        for action in invalid:
            with self.assertRaises(ValueError):
                episode.step(action)
        with self.assertRaises(ValueError):
            episode.step(self.wait(episode), side='RED')
        self.assertEqual(episode.status(), before)
        self.assertEqual([episode.observe(s) for s in ('BLUE', 'RED')], observations)
        self.assertEqual(episode._session.histories, histories)
        self.assertEqual(episode._session.env.game.rng.getstate(), rng)

    def test_fog_hidden_score_and_health_noninterference(self):
        episode = self.make(public_score=False)
        game = episode._session.env.game
        with patch.object(game, 'can_observe', return_value=False):
            for side, enemy in [('BLUE', 'red'), ('RED', 'blue')]:
                before = episode.observe(side)
                game.units[enemy].blood += 5
                game.units[enemy].score += 100
                self.assertEqual(episode.observe(side), before)
                self.assertEqual(before['enemy_visible'], [])

    def test_visible_opponent_private_details_are_masked(self):
        episode = self.make(public_score=False)
        game = episode._session.env.game
        with patch.object(game, 'can_observe', return_value=True):
            before = episode.observe('BLUE')
            enemy = game.units['red']
            enemy.ammo['a5'] = 999
            enemy.detail['target'] = 'private-target'
            enemy.detail['carried'] = ['private-passenger']
            enemy.score = 88
            self.assertEqual(episode.observe('BLUE'), before)
            self.assertEqual(set(before['enemy_visible'][0]),
                             {'id', 'name', 'typeid', 'color', 'pos',
                              'hexid', 'blood', 'suppressed'})

    def test_seat_histories_are_isolated_and_reset(self):
        episode = self.make()
        histories = episode._session.histories
        self.assertIsNot(histories['BLUE'], histories['RED'])
        histories['BLUE'].append({'t': 0, 'id': 'blue-only-memory', 'pos': 20004, 'blood': 2})
        histories['RED'].append({'t': 0, 'id': 'red-only-memory', 'pos': 20010, 'blood': 3})
        blue = episode.observe('BLUE')['enemy_known']
        red = episode.observe('RED')['enemy_known']
        self.assertIn('blue-only-memory', [u['id'] for u in blue])
        self.assertNotIn('red-only-memory', [u['id'] for u in blue])
        self.assertIn('red-only-memory', [u['id'] for u in red])
        self.assertNotIn('blue-only-memory', [u['id'] for u in red])
        episode.reset(seed=17)
        self.assertNotIn('blue-only-memory', [u['id'] for u in episode.observe('BLUE')['enemy_known']])

    def test_deterministic_reset_without_global_rng_changes(self):
        episode = self.make(max_seconds=30)
        global_rng = random.getstate()
        def rollout():
            result = [episode.reset(seed=29)]
            while episode.current_actor:
                result.append(episode.observe(episode.current_actor))
                result.append(episode.step(self.wait(episode)))
            return result
        self.assertEqual(rollout(), rollout())
        self.assertEqual(random.getstate(), global_rng)

    def test_public_scoreboard_and_unequal_terminal_totals(self):
        path = self.snapshot / 'units.json'
        units = json.loads(path.read_text())
        units[0]['score'], units[1]['score'] = 2, 5
        path.write_text(json.dumps(units))
        episode = self.make(max_seconds=10)
        # score 改成 2 / 5，blood 仍为 4 ⇒ 8 / 20
        self.assertEqual(episode.observe('BLUE')['score'], {'own': 8, 'enemy': 20, 'diff': -12})
        self.assertEqual(episode.observe('RED')['score'], {'own': 20, 'enemy': 8, 'diff': 12})
        self.assertEqual(episode.status()['rewards'], {'BLUE': 0, 'RED': 0})
        state = episode.step(self.wait(episode))
        self.assertEqual(state['rewards'], {'BLUE': 8, 'RED': 20})
        self.assertEqual(state['winner'], 'RED')
        self.assertNotIn('score', self.make(public_score=False).observe('BLUE'))

    def test_hidden_occupancy_does_not_change_choices_and_failed_intent_costs_time(self):
        from engine.game import Game
        (self.snapshot / 'cities.json').write_text(json.dumps([
            {'id': 'objective', 'pos': 20004, 'flag': None}]))
        path = self.snapshot / 'units.json'
        units = json.loads(path.read_text())
        with patch.object(Game, 'can_observe', return_value=False):
            far = self.make(max_seconds=60, public_score=False)
            units[1]['pos'] = 20006
            path.write_text(json.dumps(units))
            near = self.make(max_seconds=60, public_score=False)
            self.assertEqual(far.observe('BLUE'), near.observe('BLUE'))
            self.assertEqual(far.legal_actions(), near.legal_actions())
            action = next(a for a in near.legal_actions() if a['type'] == 'capture')
            state = near.step(action)
            self.assertEqual(state['seconds'], 30)
            self.assertEqual(state['steps'], 1)
            self.assertEqual(state['current_actor'], 'RED')
            self.assertEqual(state['rewards'], {'BLUE': 0, 'RED': 0})
            self.assertNotIn('info', state)
            self.assertIsNone(near.observe('BLUE')['objectives'][0]['held_by'])
            far.step(action)
            self.assertEqual(far.observe('BLUE')['objectives'][0]['held_by'], 'BLUE')

    def test_reset_terminal_payout_and_opponent_capture_endpoint(self):
        episode = TwoPlayerEpisode(self.snapshot, max_seconds=0)
        self.assertEqual(episode.reset(seed=1)['rewards'], {'BLUE': 4, 'RED': 4})
        self.assertEqual(episode.status()['rewards'], {'BLUE': 0, 'RED': 0})
        (self.snapshot / 'cities.json').write_text(json.dumps([
            {'id': 'objective', 'pos': 20004, 'flag': 'RED'}]))
        episode = self.make(victory='capture')
        self.assertEqual(episode.status()['reason'], 'all_objectives_owned')
        self.assertEqual(episode.status()['winner'], 'RED')
        self.assertTrue(episode.observe('BLUE')['done'])
        self.assertTrue(episode.observe('RED')['done'])

    def test_interleaved_independent_episodes(self):
        a, b = self.make(max_seconds=30), self.make(max_seconds=30)
        for _ in range(3):
            a_result = a.step(self.wait(a))
            self.assertEqual(b.status()['steps'], a_result['steps'] - 1)
            b_result = b.step(self.wait(b))
            self.assertEqual(a_result, b_result)
        self.assertIsNot(a._session.histories['BLUE'], b._session.histories['BLUE'])
        self.assertIsNot(a._session.env.game.rng, b._session.env.game.rng)

    def test_configuration_and_seat_validation(self):
        for kwargs in ({'max_seconds': -1}, {'max_steps': True}, {'dt': 0},
                       {'move_radius': -1}, {'victory': 'custom'}):
            with self.assertRaises(ValueError):
                TwoPlayerEpisode(self.snapshot, **kwargs)
        episode = self.make()
        for seed in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                episode.reset(seed=seed)
        for call in (lambda: episode.observe('GREEN'),
                     lambda: episode.legal_actions('GREEN')):
            with self.assertRaises(ValueError):
                call()


if __name__ == '__main__':
    unittest.main()
