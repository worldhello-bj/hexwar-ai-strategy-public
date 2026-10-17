"""Per-unit occupancy must apply equally to individual and group commands."""
import json
import unittest
from engine import geometry as G
from env.realtime_api import RealtimeEpisode
from tests.test_missile_lifecycle import MissileFixture


class RealtimeOccupancyTests(MissileFixture):
    def episode(self):
        e = RealtimeEpisode(str(self.root), max_seconds=300, move_radius=1,
                            deploy_window='any')
        e.reset(seed=3)
        return e

    def test_busy_unit_cannot_move_but_other_unit_can(self):
        e = self.episode()
        e._session.env.game.units['carrier'].detail['busy_until'] = 10
        actions = e.legal_actions()
        self.assertFalse(any(a['type']=='move' and a['unit']=='carrier' for a in actions))
        self.assertTrue(any(a['type']=='move' and a['unit']=='squad' for a in actions))
        self.assertIn({'type':'keep'}, actions)

    def test_launch_checks_carrier_and_missile_occupancy(self):
        e = self.episode(); g = e._session.env.game
        self.assertTrue(g.deploy_missile('carrier', 'missileA')[0])
        for uid in ('carrier', 'missileA'):
            with self.subTest(uid=uid):
                g.units[uid].detail['busy_until'] = 10
                e._legal_key = e._obs_key = None
                self.assertFalse(any(a['type']=='launch' for a in e.legal_actions()))
                g.units[uid].detail['busy_until'] = 0
        e._legal_key = e._obs_key = None
        self.assertTrue(any(a['type']=='launch' for a in e.legal_actions()))

    def test_actual_launch_at30_blocks_both_units_until40(self):
        e = self.episode(); g = e._session.env.game
        e.step({'type':'load', 'unit':'missileA', 'carrier':'carrier'})
        while g.second < 30: e.step({'type':'keep'})
        e.step({'type':'launch', 'unit':'carrier', 'missile':'missileA'})
        e.step({'type':'keep'})
        self.assertEqual(g.second, 31)
        for uid in ('carrier', 'missileA'):
            self.assertEqual(g.units[uid].detail['busy_until'], 40)
            self.assertFalse(any(a['type']=='move' and a['unit']==uid
                                 for a in e.legal_actions()))
        while g.second < 40: e.step({'type':'keep'})
        self.assertTrue(any(a['type']=='move' and a['unit']=='missileA'
                            for a in e.legal_actions()))

    def test_active_leg_blocks_retarget_and_stop_cancels_destination(self):
        e = self.episode(); g = e._session.env.game
        move = next(a for a in e.legal_actions()
                    if a['type']=='move' and a['unit']=='carrier')
        origin = g.units['carrier'].pos
        e.step(move); e.step({'type':'keep'})
        u = g.units['carrier']; first_deadline = u.detail['move_until']
        self.assertEqual(u.pos, origin)
        self.assertFalse(any(a['type']=='move' and a['unit']=='carrier'
                             for a in e.legal_actions()))
        e.step({'type':'stop', 'unit':'carrier'})
        self.assertNotIn('move_to', u.detail)
        self.assertNotIn('move_until', u.detail)
        self.assertEqual(u.pos, origin)
        e.step({'type':'keep'})
        e.step(move)
        self.assertEqual(u.detail['move_until'], first_deadline + 2)

    def group_episode(self):
        units = [dict(id=uid, data=400, typeid=2, color=side, blood=4,
                      score=10, pos=G.make_mapid(10,col))
                 for uid,side,col in [('a','BLUE',10),('b','BLUE',11),('foe','RED',13)]]
        (self.root/'units.json').write_text(json.dumps(units))
        return self.episode()

    def test_group_excludes_busy_nonprimary_participant(self):
        e = self.group_episode()
        group = next(a for a in e.legal_actions() if a['type']=='area_fire' and len(a['units'])==2)
        e._session.env.game.units[group['units'][1]].detail['busy_until'] = 10
        e._legal_key = None
        self.assertNotIn(group, e.legal_actions())
        with self.assertRaises(ValueError): e.step(group)

    def test_group_occupies_every_participant_and_expires(self):
        e = self.group_episode(); g = e._session.env.game
        group = next(a for a in e.legal_actions() if a['type']=='area_fire' and len(a['units'])==2)
        e.step(group)
        for uid in group['units']:
            self.assertEqual(g.units[uid].detail.get('busy_until'), 10)
        while g.second < 10: e.step({'type':'keep'})
        actions = e.legal_actions()
        for uid in group['units']:
            self.assertTrue(any(a['type']=='move' and a['unit']==uid for a in actions))

if __name__ == '__main__': unittest.main()
