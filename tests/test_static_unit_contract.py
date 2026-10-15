"""Static fortifications cannot manufacture movement or capture rewards."""
import json
from pathlib import Path
import tempfile
import unittest

from engine import Game, Unit, scoring
from engine import geometry as G
from env.hexwar_env import HexwarEnv


class StaticUnitContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.start, self.goal = 20004, 20006
        fixtures = {
            'map_hexes.json': {str(p): {'step': 1, 'cond': 0, 'elevation': 0}
                              for p in (20004, 20006, 20008, 20010, 20012)},
            'cities.json': [{'id': 'obj', 'pos': self.goal, 'flag': None, 'name': '主要'}],
            'units.json': [dict(id='bunker', data=408, typeid=4, color='BLUE',
                               pos=self.start, blood=1, score=0, step=0),
                           dict(id='enemy', typeid=1, color='RED', pos=20012,
                                blood=1, score=1)],
        }
        for name, value in fixtures.items():
            (self.root / name).write_text(json.dumps(value))
        self.game = Game(snapshot=self.root)

    def test_bunker_step_and_commit_are_rejected(self):
        u = self.game.units['bunker']
        self.assertEqual(self.game.step_seconds(u, self.start, self.goal)[0], 0)
        self.assertEqual(self.game.commit_step(u, self.start, self.goal)[0], 0)

    def test_bunker_cannot_register_or_resume_queued_move(self):
        self.assertFalse(self.game.begin_move('bunker', [self.goal])[0])
        u = self.game.units['bunker']
        u.detail.update(path=[self.goal], path_idx=0)
        self.game.advance(1000)
        self.assertEqual(u.pos, self.start)
        self.assertNotIn('path', u.detail)

    def test_bunker_cannot_directly_capture_even_if_at_objective(self):
        u = self.game.units['bunker']
        u.pos = self.goal
        self.assertIsNone(self.game.apply_city_capture(u))
        self.assertIsNone(self.game.cities[0]['flag'])

    def test_env_rejects_teleport_capture_and_excludes_actions(self):
        for deep in (False, True):
            env = HexwarEnv(snapshot=self.root, max_seconds=2000)
            env.reset(seed=1)
            self.assertFalse(any(a['type'] == 'move' for a in env.legal_actions(deep_validate=deep)))
            before = env.score()
            _, _, _, info = env.step({'type': 'move', 'unit': 'bunker',
                                      'path': [self.start, self.goal]})
            self.assertTrue(info.get('invalid_action'))
            self.assertEqual(env.game.units['bunker'].pos, self.start)
            self.assertEqual(env.game.second, 0)
            self.assertEqual(env.score(), before)
            env.game.units['bunker'].pos = self.goal
            self.assertFalse(any(a['type'] == 'capture' for a in env.legal_actions(deep_validate=deep)))
            _, _, _, info = env.step({'type': 'capture', 'unit': 'bunker'})
            self.assertTrue(info.get('invalid_action'))
            self.assertIsNone(env.game.cities[0]['flag'])

    def test_all_type_mobility_and_capture_contracts(self):
        for typeid in (1, 2, 3, 4, 99):
            u = Unit(dict(id='unit', typeid=typeid, blood=1, color='BLUE',
                          pos=self.start, step=0))
            self.assertEqual(self.game.step_seconds(u, self.start, self.goal)[0] > 0,
                             typeid in (1, 2, 3))
            self.game.cities[0]['flag'] = None
            u.pos = self.goal
            self.assertEqual(self.game.apply_city_capture(u) is not None,
                             typeid in (1, 2))
        u = Unit(dict(id='artillery', typeid=2, blood=1, color='BLUE',
                      pos=self.goal, name='榴弹炮'))
        self.assertIsNone(self.game.apply_city_capture(u))

    def test_competition_bunkers_are_all_static_and_opening_unchanged(self):
        game = Game(snapshot=Path(__file__).resolve().parents[1] / 'scenario/7527')
        bunkers = [u for u in game.units.values() if u.typeid == 4]
        self.assertEqual(len(bunkers), 3)
        self.assertEqual(sum(u.is_mobile() for u in game.units.values()), 33)
        self.assertEqual(sum(game.capture_eligible(u) for u in game.units.values()), 24)
        self.assertIn('Obj_100170', [u.id for u in bunkers])
        for u in bunkers:
            for nb in G.neighbors(u.pos):
                self.assertEqual(game.step_seconds(u, u.pos, nb)[0], 0)
        scores, _ = scoring.both_scores(game)
        self.assertEqual({s: x['total'] for s, x in scores.items()}, {'BLUE': 234, 'RED': 354})


if __name__ == '__main__':
    unittest.main()
