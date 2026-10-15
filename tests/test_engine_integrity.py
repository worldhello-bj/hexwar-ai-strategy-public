"""Offline state-integrity regressions using a tiny synthetic board."""
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.game import Game, Unit
from engine import crt


class EngineIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        fixtures = {
            'map_hexes.json': {str(p): {'step': 1, 'cond': 0, 'elevation': 0}
                               for p in (20004, 20006, 20008)},
            'cities.json': [{'pos': 20004, 'flag': 'BLUE', 'name': 'Token'}],
            'units.json': [dict(id='a', name='A', color='RED', typeid=1,
                                pos=20004, blood=4, wz='29,144', a5=99),
                           dict(id='b', name='B', color='BLUE', typeid=1,
                                pos=20006, blood=4, wz='29', a5=99)],
        }
        for filename, data in fixtures.items():
            (self.path / filename).write_text(json.dumps(data), encoding='utf-8')
        self.game = Game(seed=17, snapshot=self.path)

    def test_seed_does_not_modify_global_random(self):
        random.seed(991)
        before = random.getstate()
        Game(seed=17, snapshot=self.path)
        self.assertEqual(before, random.getstate())

    def test_interleaved_games_have_independent_random_sequences(self):
        a, b = (Game(seed=17, snapshot=self.path) for _ in range(2))
        rolls = [[], []]
        for _ in range(8):
            for i, g in enumerate((a, b)):
                g.units['b'].blood = 100
                result, error = g.attack('a', 'b')
                self.assertIsNone(error)
                rolls[i].append(result['roll'])
        self.assertEqual(*rolls)

    def test_step_query_is_pure(self):
        u = self.game.units['a']
        u.ste = 4
        before = (copy.deepcopy(u.detail), u.fatigued_until, list(self.game.log))
        self.assertGreater(self.game.step_seconds(u, u.pos, 20006)[0], 0)
        self.assertEqual(before, (u.detail, u.fatigued_until, self.game.log))

    def test_commit_step_applies_cost_only_after_validation(self):
        u = self.game.units['a']
        u.ste = 4
        self.assertEqual(self.game.commit_step(u, u.pos, 20008)[0], 0)
        self.assertEqual(u.fatigued_until, 0)
        self.assertEqual(self.game.log, [])
        self.assertGreater(self.game.commit_step(u, u.pos, 20006)[0], 0)
        self.assertTrue(u.fatigued(self.game.second))
        self.assertEqual(u.pos, 20004)  # Position belongs to the caller.
        self.assertEqual(self.game.commit_step(u, u.pos, 20006)[0], 0)
        self.assertEqual(len(self.game.log), 1)

    def test_rejected_stacking_does_not_apply_cost(self):
        u = self.game.units['a']
        u.ste = 4
        for i in range(4):
            v = Unit(dict(id=str(i), typeid=1, color='RED', blood=1, pos=20006))
            self.game.units[v.id] = v
        self.assertEqual(self.game.commit_step(u, u.pos, 20006)[0], 0)
        self.assertEqual((u.fatigued_until, u.detail, self.game.log), (0, {}, []))

    def test_nonadjacent_or_inactive_step_is_rejected(self):
        u = self.game.units['a']
        self.assertEqual(self.game.step_seconds(u, u.pos, 20008)[0], 0)
        self.assertEqual(self.game.step_seconds(u, u.pos, u.pos)[0], 0)
        u.blood = 0
        self.assertEqual(self.game.step_seconds(u, u.pos, 20006)[0], 0)
        u.blood, u.detail = 4, {'carrier': 'token'}
        self.assertEqual(self.game.step_seconds(u, u.pos, 20006)[0], 0)

    def test_distance_override_cannot_bypass_board(self):
        for dist in (-1, 0, 0.5, float('nan')):
            result, error = self.game.attack('a', 'b', dist=dist)
            self.assertIsNone(result)
            self.assertIsNotNone(error)
        self.assertEqual(self.game.units['a'].ammo['a5'], 99)

    def test_grenade_randomness_does_not_touch_global_stream(self):
        before = random.getstate()
        self.game.grenade('a', 20006)
        self.assertEqual(before, random.getstate())

    def test_zero_score_is_preserved(self):
        with patch('engine.game.EQUIPMENT', {999: ('Token', 1, '', 7, (0, 0, 0), [])}):
            self.assertEqual(Unit({'id': 'x', 'data': 999, 'score': 0}).score, 0)

    def test_no_resource_means_no_grenade(self):
        a = self.game.units['a']
        a.ammo['a5'] = 0
        before = copy.deepcopy(a.ammo)
        result, error = self.game.grenade('a', 20006)
        self.assertIsNone(result)
        self.assertIsNotNone(error)
        self.assertEqual(a.ammo, before)
        self.assertIn(144, a.wz)

    def test_same_team_rejected_in_other_cell(self):
        self.game.units['b'].color = 'RED'
        result, error = self.game.attack('a', 'b')
        self.assertIsNone(result)
        self.assertIsNotNone(error)
        self.assertEqual(self.game.units['a'].ammo['a5'], 99)

    def test_dead_or_carried_unit_cannot_capture(self):
        a = self.game.units['a']
        for blood, detail in ((0, {}), (4, {'carrier': 'token'})):
            a.blood, a.detail = blood, detail
            self.game.units['b'].blood = 0
            self.assertIsNone(self.game.apply_city_capture(a))
            self.assertEqual(self.game.cities[0]['flag'], 'BLUE')

    def test_nonfinite_clock_rejected_without_mutation(self):
        for seconds in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(seconds=seconds):
                with self.assertRaises(ValueError):
                    self.game.advance(seconds)
                self.assertEqual(self.game.second, 0)

    def test_invalid_crt_distances_rejected(self):
        for distance in (-1, 0.5, float('nan')):
            with self.subTest(distance=distance):
                with self.assertRaises(ValueError):
                    crt.direct_fire({'units': 1}, {'typeid': 1}, 'Token', distance)


if __name__ == '__main__':
    unittest.main()
