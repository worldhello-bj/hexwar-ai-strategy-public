"""Transport's existing 30-second cost applies to every participating unit."""
import unittest
import json
from tests.test_missile_lifecycle import MissileFixture
from env.realtime_api import RealtimeEpisode

class TransportOccupancyTests(MissileFixture):
    def episode(self):
        e = RealtimeEpisode(str(self.root), max_seconds=120, move_radius=1,
                            deploy_window='any')
        e.reset(seed=11)
        return e

    def test_load_occupies_carrier_and_passenger_and_blocks_early_unload(self):
        e=self.episode(); g=e._session.env.game
        e.step({'type':'load','unit':'squad','carrier':'carrier'})
        e.step({'type':'keep'})
        self.assertEqual(g.second, 1)
        self.assertEqual(g.units['carrier'].detail.get('busy_until'),30)
        self.assertEqual(g.units['squad'].detail.get('busy_until'),30)
        self.assertNotIn({'type':'unload','unit':'carrier'},e.legal_actions())
        while g.second<30: e.step({'type':'keep'})
        self.assertIn({'type':'unload','unit':'carrier'},e.legal_actions())

    def test_load_rejects_busy_carrier_without_blocking_other_units(self):
        e=self.episode(); g=e._session.env.game
        g.units['carrier'].detail['busy_until']=10
        self.assertNotIn({'type':'load','unit':'squad','carrier':'carrier'},e.legal_actions())
        self.assertTrue(any(a['type']=='move' and a['unit']=='squad' for a in e.legal_actions()))

    def loaded_episode(self):
        e=self.episode(); e.step({'type':'load','unit':'squad','carrier':'carrier'})
        while e._session.env.game.second<30: e.step({'type':'keep'})
        return e

    def test_unload_checks_busy_passenger_even_when_carrier_ready(self):
        e=self.loaded_episode();g=e._session.env.game
        g.units['squad'].detail['busy_until']=40
        e._legal_key=None
        self.assertNotIn({'type':'unload','unit':'carrier'},e.legal_actions())

    def test_unload_occupies_carrier_and_disembarked_passenger(self):
        e=self.loaded_episode();g=e._session.env.game
        e.step({'type':'unload','unit':'carrier'})
        for uid in ('carrier','squad'):
            self.assertEqual(g.units[uid].detail.get('busy_until'),60)
        self.assertNotIn('carrier',g.units['squad'].detail)
        e.step({'type':'keep'})
        self.assertFalse(any(a['type']=='move' and a['unit'] in ('carrier','squad') for a in e.legal_actions()))
        while g.second<60:e.step({'type':'keep'})
        self.assertTrue(any(a['type']=='move' and a['unit']=='squad' for a in e.legal_actions()))

    def test_unload_occupies_all_passengers_but_not_carried_missile(self):
        path=self.root/'units.json'; units=json.loads(path.read_text())
        squad=next(u for u in units if u['id']=='squad')
        units.append(dict(squad,id='squad2')); path.write_text(json.dumps(units))
        e=self.episode();g=e._session.env.game
        for index,uid in enumerate(('squad','squad2','missileA')):
            e.step({'type':'load','unit':uid,'carrier':'carrier'})
            while g.second<(index+1)*30:e.step({'type':'keep'})
        e.step({'type':'unload','unit':'carrier'})
        for uid in ('carrier','squad','squad2'):
            self.assertEqual(g.units[uid].detail['busy_until'],120)
        self.assertEqual(g.units['missileA'].detail['busy_until'],90)
        self.assertEqual(g.units['missileA'].detail['carrier'],'carrier')

if __name__=='__main__':unittest.main()
