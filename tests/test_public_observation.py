"""Public-board contract tests using original synthetic abstract-token fixtures."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from engine import geometry as G
from engine.game import Game
from env.episode_api import TwoPlayerEpisode
from env.match_eval import write_synthetic_snapshot
from env.public_observation import PublicObservationAdapter


class PublicObservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.snapshot = write_synthetic_snapshot(Path(self.tmp.name) / 'tokens')

    def make(self, **kwargs):
        episode = TwoPlayerEpisode(self.snapshot, **kwargs)
        adapter = PublicObservationAdapter(episode)
        adapter.reset(seed=17)
        return episode, adapter

    def test_full_board_whitelist_coordinates_and_detachment(self):
        path = self.snapshot / 'map_hexes.json'
        raw = json.loads(path.read_text())
        raw['30021'] = {'elevation': 4, 'cond': 8, 'step': -1,
                        'secret': 'never-export', 'occupants': ['hidden-red']}
        path.write_text(json.dumps(raw))
        episode, adapter = self.make()
        obs = adapter.observe('BLUE')
        self.assertEqual(obs['schema_version'], 2)
        self.assertEqual(set(obs['board']['cells']), set(raw))
        for mapid, cell in obs['board']['cells'].items():
            self.assertEqual(set(cell), {'id', 'row', 'col', 'elevation',
                                        'terrain_code', 'step_code'})
            self.assertEqual((cell['row'], cell['col']), G.split_mapid(mapid))
            self.assertEqual(str(G.make_mapid(cell['row'], cell['col'])), mapid)
        self.assertEqual(obs['board']['cells']['30021']['step_code'], '-1')
        before = adapter.observe('BLUE')
        obs['board']['cells']['30021']['elevation'] = 999
        obs['own']['blue']['blood'] = 999
        obs['own']['blue']['position']['row'] = 999
        self.assertEqual(adapter.observe('BLUE'), before)
        self.assertEqual(episode._session.env.game.hexes[30021]['elevation'], 4)
        json.dumps(before, allow_nan=False)

    def test_odd_even_roundtrip_and_neighbors(self):
        for row in (4, 5, 6, 7):
            for col in (2, 3, 10):
                mapid = G.make_mapid(row, col)
                self.assertEqual(G.split_mapid(mapid), (row, col))
                neighbors = G.neighbors(mapid)
                self.assertEqual(len(neighbors), 6)
                for other in neighbors:
                    self.assertEqual(G.distance(mapid, other), 1)
                    self.assertIn(mapid, G.neighbors(other))
        _, adapter = self.make()
        obs = adapter.observe('BLUE')
        context = adapter._candidate_context({'path': [20004, 20005, 30004]}, obs)
        self.assertEqual([(p['row'], p['col']) for p in context['path']],
                         [(4, 2), (5, 2), (6, 2)])

    def test_hidden_perturbation_cannot_affect_public_view_or_candidates(self):
        with patch.object(Game, 'can_observe', return_value=False):
            episode, adapter = self.make()
            before, choices = adapter.observe('BLUE'), adapter.candidate_views('BLUE')
            enemy = episode._session.env.game.units['red']
            enemy.pos = 20008
            # 注：这里**不能**再扰动 `blood` —— 它已通过公开的「剩余兵力分」合计反出（平台行为），
            #     见 test_public_score_reveals_only_aggregate_enemy_strength。
            enemy.ammo['a5'] = 999
            enemy.detail.update(target='secret', carrier='secret')
            enemy.stunned_until = 12345
            enemy.fatigued_until = 54321
            self.assertEqual(adapter.observe('BLUE'), before)
            self.assertEqual(adapter.candidate_views('BLUE'), choices)
            self.assertEqual(before['enemy_visible'], {})
            self.assertEqual(before['score'], {'own': 4, 'enemy': 4, 'diff': 0})

    def test_public_score_reveals_only_aggregate_enemy_strength(self):
        """血量只会以「剩余兵力分合计」形式公开，不会连带泄\u6f0f逐单位字段。

        平台计分页本身就公开「剩余兵力分」，所以敌人血量变化**合法地**反在总量里；
        这是刻意的信息隔离边界，见 `docs/计分规则待确认.md`。
        """
        with patch.object(Game, 'can_observe', return_value=False):
            episode, adapter = self.make()
            before = adapter.observe('BLUE')
            self.assertEqual(before['score'], {'own': 4, 'enemy': 4, 'diff': 0})
            episode._session.env.game.units['red'].blood = 9
            after = adapter.observe('BLUE')
            # 敌血 4→9，且 score=1 ⇒ 敌方合计 4→9
            self.assertEqual(after['score'], {'own': 4, 'enemy': 9, 'diff': -5})
            self.assertEqual(after['enemy_visible'], {})
            left, right = dict(before), dict(after)
            left.pop('score'), right.pop('score')
            self.assertEqual(right, left)          # 除 score 外逐字段不变

    def test_visible_private_fields_and_own_expiry_are_absent(self):
        with patch.object(Game, 'can_observe', return_value=True):
            episode, adapter = self.make()
            before = adapter.observe('BLUE')
            game = episode._session.env.game
            game.units['red'].ammo['a5'] = 999
            game.units['red'].detail['target'] = 'secret'
            game.units['blue'].fatigued_until = 999
            self.assertEqual(adapter.observe('BLUE'), before)
            self.assertEqual(set(before['enemy_visible']['red']),
                             {'id', 'name', 'typeid', 'color', 'pos', 'position',
                              'hexid', 'blood', 'suppressed'})
            self.assertNotIn('fatigued_until', before['own']['blue'])

    def test_distant_public_terrain_is_visible_without_local_action_change(self):
        path = self.snapshot / 'map_hexes.json'
        raw = json.loads(path.read_text())
        raw['90040'] = {'elevation': 2, 'cond': 0, 'step': 1}
        path.write_text(json.dumps(raw))
        _, adapter = self.make()
        before, choices = adapter.observe('BLUE'), adapter.legal_actions()
        raw['90040']['cond'] = 8
        path.write_text(json.dumps(raw))
        _, changed = self.make()
        self.assertNotEqual(changed.observe('BLUE'), before)
        self.assertEqual(changed.observe('BLUE')['board']['cells']['90040']['terrain_code'], '8')
        self.assertEqual(changed.legal_actions(), choices)

    def test_entity_order_invariance_and_path_order_sensitivity(self):
        episode, adapter = self.make()
        base = episode.observe('BLUE')
        extra = copy.deepcopy(base['own'][0])
        extra['id'] = 'other'
        base['own'].append(extra)
        with patch.object(episode, 'observe', return_value=base):
            before = adapter.observe('BLUE')
            base['own'].reverse()
            self.assertEqual(adapter.observe('BLUE'), before)
        forward = adapter._candidate_context({'path': [20004, 20006, 20008]}, before)
        reverse = adapter._candidate_context({'path': [20008, 20006, 20004]}, before)
        self.assertNotEqual(forward, reverse)
        self.assertEqual(forward['path'], list(reversed(reverse['path'])))

    def test_stale_memory_and_objectives_keep_public_coordinates(self):
        (self.snapshot / 'cities.json').write_text(json.dumps([
            {'id': 'objective', 'pos': 20006, 'name': 'Public', 'flag': None}]))
        with patch.object(Game, 'can_observe', return_value=False):
            episode, adapter = self.make()
            episode._session.histories['BLUE'].append(
                {'id': 'last-seen', 'pos': 20008, 't': 0, 'blood': 2})
            obs = adapter.observe('BLUE')
            self.assertEqual(obs['enemy_known']['last-seen']['position'],
                             {'id': '20008', 'row': 4, 'col': 4})
            self.assertTrue(obs['enemy_known']['last-seen']['stale'])
            self.assertEqual(obs['objectives']['objective']['position']['id'], '20006')
            self.assertNotIn('last-seen', adapter.observe('RED')['enemy_known'])

    def test_actions_untouched_identity_context_detachment_and_validity(self):
        episode, adapter = self.make()
        actions = episode.legal_actions('BLUE')
        original = copy.deepcopy(actions)
        with patch.object(episode, 'legal_actions', return_value=actions):
            views = adapter.candidate_views('BLUE')
        self.assertEqual(actions, original)
        for action, view in zip(actions, views):
            self.assertIs(view['action'], action)
        move = next(v for v in views if v['action']['type'] == 'move')
        move['public_context']['path'][0]['id'] = 'changed'
        self.assertEqual(actions, original)
        self.assertNotEqual(adapter.observe('BLUE')['own']['blue']['pos'], 'changed')
        with patch.object(episode, 'step', wraps=episode.step) as step:
            adapter.step(move['action'], side='BLUE')
            self.assertIs(step.call_args.args[0], move['action'])
        self.assertEqual(adapter.status()['steps'], 1)

    def test_score_option_zero_intermediate_and_exact_terminal_totals(self):
        path = self.snapshot / 'units.json'
        units = json.loads(path.read_text())
        units[0]['score'], units[1]['score'] = 2, 5   # blood 仍为 4 ⇒ 8 / 20
        path.write_text(json.dumps(units))
        _, adapter = self.make(max_seconds=20)
        self.assertEqual(adapter.observe('BLUE')['score'], {'own': 8, 'enemy': 20, 'diff': -12})
        for terminal in (False, True):
            wait = next(v['action'] for v in adapter.candidate_views()
                        if v['action']['type'] == 'wait')
            state = adapter.step(wait)
            self.assertEqual(state['terminated'], terminal)
            self.assertEqual(state['rewards'], {'BLUE': 8, 'RED': 20} if terminal
                             else {'BLUE': 0, 'RED': 0})
        self.assertEqual(adapter.status()['rewards'], {'BLUE': 0, 'RED': 0})
        self.assertEqual(adapter.candidate_views(), [])
        _, private = self.make(public_score=False)
        self.assertNotIn('score', private.observe('BLUE'))

    def test_lifecycle_validation_and_truncation_delegate(self):
        adapter = PublicObservationAdapter(TwoPlayerEpisode(self.snapshot, max_steps=1))
        with self.assertRaises(RuntimeError):
            adapter.observe('BLUE')
        adapter.reset(seed=17)
        with self.assertRaises(ValueError):
            adapter.observe('GREEN')
        self.assertEqual(adapter.candidate_views('RED'), [])
        state = adapter.step(next(a for a in adapter.legal_actions() if a['type'] == 'wait'))
        self.assertTrue(state['truncated'])
        self.assertIsNone(state['final_scores'])
        self.assertEqual(state['rewards'], {'BLUE': 0, 'RED': 0})


if __name__ == '__main__':
    unittest.main()
