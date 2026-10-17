"""A legally killed mover must disappear from the next seat's action menu."""
import json
import tempfile
import unittest
from pathlib import Path
from engine import geometry as G
from env.realtime_api import RealtimeEpisode

class DeadUnitMenuTests(unittest.TestCase):
    def test_killed_mover_has_no_stop_action(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            origin,target,destination,far=[G.make_mapid(r,c) for r,c in ((10,10),(10,12),(10,13),(30,30))]
            cells={str(G.make_mapid(r,c)):{'cond':0,'step':1,'elevation':0}
                   for r in range(7,15) for c in range(7,17)}
            cells[str(far)]={'cond':0,'step':1,'elevation':0}
            units=[dict(id='shooter',data=400,typeid=2,color='BLUE',blood=4,pos=origin,a4=20,a5=20,a3=20),
                   dict(id='victim',data=382,typeid=1,color='RED',blood=1,pos=target),
                   dict(id='survivor',data=0,typeid=1,color='RED',blood=1,pos=far)]
            for name,value in [('map_hexes.json',cells),('units.json',units),('cities.json',[])]:
                (p/name).write_text(json.dumps(value))
            e=RealtimeEpisode(str(p),max_seconds=100,move_radius=1,fog=False)
            e.reset(seed=2); e.step({'type':'keep'})
            move=next(a for a in e.legal_actions() if a['type']=='move' and a['unit']=='victim' and a['path'][-1]==destination)
            e.step(move)
            attack=next(a for a in e.legal_actions() if a['type']=='attack' and a['unit']=='shooter' and a['target']=='victim' and a['weapon']==43 and not a.get('track'))
            e.step(attack)
            self.assertEqual(e._session.env.game.units['victim'].blood,0)
            self.assertEqual(e.current_actor,'RED')
            self.assertNotIn('victim',[u['id'] for u in e.observe('RED')['own']])
            self.assertFalse(any(a.get('unit')=='victim' for a in e.legal_actions()))
            with self.assertRaises(ValueError):e.step({'type':'stop','unit':'victim'})

if __name__=='__main__':unittest.main()
