"""Regression for timed arrival without changing movement/capture rules."""
import copy,json,tempfile,unittest
from pathlib import Path
from engine.game import Game
from engine import geometry as G

class TimedArrivalIntegrityTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.start=G.make_mapid(10,10);self.dest=G.make_mapid(10,11);self.next=G.make_mapid(10,12)
  cells={str(G.make_mapid(r,c)):{'cond':0,'elevation':0,'step':1} for r in range(7,16) for c in range(7,20)}
  units=[dict(id='car',data=396,typeid=2,color='BLUE',blood=4,pos=self.start,a5=4),dict(id='passenger',data=382,typeid=1,color='BLUE',blood=3,pos=self.start,a5=1),dict(id='enemy',data=400,typeid=2,color='RED',blood=3,pos=G.make_mapid(15,19))]
  for name,data in [('map_hexes.json',cells),('units.json',units),('cities.json',[dict(id='city',name='city',pos=self.dest,flag='RED',ico='RED')])]:Path(self.tmp.name,name).write_text(json.dumps(data))
 def game(self):
  g=Game(snapshot=self.tmp.name,seed=3);g.start();return g
 def test_arrival_waits_exact_cost(self):
  g=self.game();u=g.units['car'];cost=g.step_seconds(u,u.pos,self.dest)[0]
  self.assertTrue(g.begin_move(u.id,[self.dest])[0]);self.assertEqual(u.pos,self.start)
  self.assertEqual(u.detail['move_until'],cost)
  g.advance(int(cost)-1);self.assertEqual(u.pos,self.start)
  g.advance(1);self.assertEqual(u.pos,self.dest);self.assertNotIn(u.id,g.pending_moves())
 def test_retarget_cannot_reset_active_deadline(self):
  g=self.game();u=g.units['car'];g.begin_move(u.id,[self.dest]);g.advance(1)
  before=copy.deepcopy(u.detail);self.assertFalse(g.begin_move(u.id,[self.next])[0]);self.assertEqual(u.detail,before)
 def test_passenger_sync_and_capture_at_arrival_only(self):
  g=self.game();u=g.units['car'];p=g.units['passenger'];p.detail['carrier']=u.id;u.detail['carried']=[p.id]
  g.begin_move(u.id,[self.dest]);due=u.detail['move_until'];g.advance(int(due)-1)
  self.assertEqual(p.pos,self.start);self.assertEqual(g.cities[0]['flag'],'RED')
  g.advance(1);self.assertEqual((p.pos,p.pos2),(self.dest,self.dest));self.assertEqual(g.cities[0]['flag'],'BLUE')
 def test_contested_arrival_does_not_capture(self):
  g=self.game();u=g.units['car'];g.units['enemy'].pos=self.next
  g.begin_move(u.id,[self.dest]);g.advance(int(u.detail['move_until']))
  self.assertEqual(u.pos,self.dest);self.assertEqual(g.cities[0]['flag'],'RED')
 def test_cancel_does_not_teleport_and_fresh_order_pays_full_cost(self):
  g=self.game();u=g.units['car'];cost=g.step_seconds(u,u.pos,self.dest)[0];g.begin_move(u.id,[self.dest]);g.advance(1)
  g.cancel_move(u.id);g.advance(int(cost)+1);self.assertEqual(u.pos,self.start);self.assertNotIn(u.id,g.pending_moves())
  now=g.second;g.begin_move(u.id,[self.dest]);self.assertEqual(u.detail['move_until'],now+cost)
 def test_suppression_finishes_leg_then_pauses_next(self):
  g=self.game();u=g.units['car'];g.begin_move(u.id,[self.dest,self.next]);due=u.detail['move_until'];u.stunned_until=due+10
  g.advance(int(due));self.assertEqual(u.pos,self.dest);self.assertNotIn('move_to',u.detail)
  g.advance(9);self.assertEqual(u.pos,self.dest)
  g.advance(1);self.assertIn('move_to',u.detail);self.assertEqual(u.pos,self.dest)
 def test_chunked_clock_deterministic(self):
  a,b=self.game(),self.game()
  for g in (a,b):g.begin_move('car',[self.dest,self.next])
  a.advance(50)
  for _ in range(50):b.advance(1)
  self.assertEqual(a.units['car'].pos,b.units['car'].pos)
  self.assertEqual(a.units['car'].detail,b.units['car'].detail);self.assertEqual(a.rng.getstate(),b.rng.getstate());self.assertEqual(a.cities,b.cities)

 def test_step_cost_committed_once_not_once_per_tick(self):
  from unittest.mock import patch
  g=self.game();u=g.units['car'];cost=g.step_seconds(u,u.pos,self.dest)[0]
  with patch.object(g,'commit_step',wraps=g.commit_step) as commit:
   self.assertTrue(g.begin_move(u.id,[self.dest])[0]);self.assertEqual(commit.call_count,1)
   g.advance(int(cost)+1);self.assertEqual(commit.call_count,1)
 def test_invalid_first_leg_is_atomic(self):
  g=self.game();u=g.units['car'];before=copy.deepcopy(u.detail)
  self.assertFalse(g.begin_move(u.id,[999999])[0]);self.assertEqual(before,u.detail)
 def test_destination_full_at_arrival_rejects_without_second_cost(self):
  from engine.game import Unit
  from unittest.mock import patch
  g=self.game();u=g.units['car']
  with patch.object(g,'commit_step',wraps=g.commit_step) as commit:
   g.begin_move(u.id,[self.dest]);due=u.detail['move_until']
   for i in range(4):
    x=Unit(dict(id='block'+str(i),data=382,typeid=1,color='BLUE',blood=3,pos=self.dest));g.units[x.id]=x
   g.advance(int(due));self.assertEqual(u.pos,self.start);self.assertEqual(commit.call_count,1)
   self.assertNotIn(u.id,g.pending_moves());self.assertEqual(g.cities[0]['flag'],'RED')
 def test_destination_becomes_impassable_before_arrival(self):
  g=self.game();u=g.units['car'];g.begin_move(u.id,[self.dest]);due=u.detail['move_until']
  g.hexes[self.dest]['step']=0;g.advance(int(due));self.assertEqual(u.pos,self.start);self.assertNotIn(u.id,g.pending_moves())
 def test_started_charge_completes_without_charging_twice(self):
  g=self.game();u=g.units['passenger'];u.ste=4;g.begin_move(u.id,[self.dest]);due=u.detail['move_until'];fatigue=u.fatigued_until
  self.assertGreater(fatigue,0);g.advance(int(due));self.assertEqual(u.pos,self.dest);self.assertEqual(u.fatigued_until,fatigue)
