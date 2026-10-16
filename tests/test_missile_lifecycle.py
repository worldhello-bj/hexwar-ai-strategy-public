# -*- coding: utf-8 -*-
"""巡飞弹生命周期回归（issue #1）—— 用**合成**小棋盘，不依赖任何线上数据。

覆盖：显式状态机（未部署 / 车内 / 在飞 / 已消耗 / 已损失）、发射门、训练接口
（动作生成→校验→执行→实时层调度）、卸载不发射、一车一枚、攻击后消耗、计时来源。

对应的可见验收报告见 ``tools/verify_missile_lifecycle.py``（同一批场景，逐条打印）。
"""
import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path

from engine import geometry as G
from engine.game import Game, LOITER_SEC_PLACEHOLDER
from env import HexwarEnv
from env.episode_api import TwoPlayerEpisode
from env.realtime_api import RealtimeEpisode


class MissileFixture(unittest.TestCase):
    """一辆车 + 两枚巡飞弹 + 一个步兵班 + 一个敌方坦克（弹与车同格、无所属关系）。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.start = G.make_mapid(10, 10)
        self.foe = G.make_mapid(10, 12)          # 距 start 2 格：在巡飞弹侦察半径内
        hexes = {str(G.make_mapid(r, c)): {"cond": 0, "elevation": 0, "step": 1}
                 for r in range(7, 15) for c in range(7, 15)}
        units = [
            dict(id="carrier", data=396, typeid=2, color="BLUE", blood=4, score=10,
                 pos=self.start, son="WS43", a5=4),
            dict(id="squad", data=382, typeid=1, color="BLUE", blood=3, score=3,
                 pos=self.start, a5=1),
            dict(id="missileA", data=389, typeid=3, color="BLUE", blood=3, score=0,
                 pos=self.start, ico="WS43", a5=1),
            dict(id="missileB", data=389, typeid=3, color="BLUE", blood=3, score=0,
                 pos=self.start, ico="WS43", a5=1),
            dict(id="foe", data=400, typeid=2, color="RED", blood=3, score=10,
                 pos=self.foe),
        ]
        for name, payload in (("map_hexes.json", hexes), ("cities.json", []),
                              ("units.json", units)):
            (self.root / name).write_text(json.dumps(payload), encoding="utf-8")

    def game(self, **kw):
        g = Game(seed=3, snapshot=str(self.root), **kw)
        g.start()
        return g

    def env(self, **kw):
        opts = dict(snapshot=str(self.root), side="BLUE", max_seconds=900,
                    fog=False, move_radius=1)
        opts.update(kw)
        env = HexwarEnv(**opts)
        env.reset(seed=3)
        return env


class MissileStateTests(MissileFixture):
    def test_explicit_state_transitions(self):
        g = self.game()
        car, m = g.units["carrier"], g.units["missileA"]
        self.assertEqual(g.missile_state(m), "unassigned")
        self.assertIsNone(g.in_flight_missile(car.id))
        self.assertTrue(g.deploy_missile(car.id, m.id)[0])
        self.assertEqual(g.missile_state(m), "carried")
        self.assertEqual((m.blood, m.color), (0, ""))
        self.assertEqual(m.establishment, 3)
        self.assertTrue(g.launch_missile(car.id, m.id)[0])
        self.assertEqual(g.missile_state(m), "loitering")
        self.assertGreater(m.blood, 0)
        self.assertEqual(m.color, "BLUE")
        self.assertEqual(car.detail.get("launched_missile"), m.id)
        self.assertEqual(car.detail.get("stop"), 1)
        self.assertEqual(m.detail.get("loiter_seconds_source"),
                         "client:parseInt(1200/RoomSpeed)")
        g.advance(g.loiter_sec)
        self.assertEqual(g.missile_state(m), "lost")
        self.assertNotIn("launched_missile", car.detail)
        self.assertEqual(car.detail.get("stop"), 0)

    def test_externally_killed_in_flight_is_lost_and_releases(self):
        g = self.game()
        car, m = g.units["carrier"], g.units["missileA"]
        g.deploy_missile(car.id, m.id)
        g.launch_missile(car.id, m.id)
        m.blood = 0
        g.advance(1)
        self.assertEqual(g.missile_state(m), "lost")
        self.assertNotIn("launched_missile", car.detail)

    def test_opening_state_is_not_silently_linked(self):
        """想定 7527 的 `car=Obj_0` 只作证据留存，不得静默建立所属关系。"""
        g = self.game()
        for unit in g.units.values():
            self.assertNotIn("car", unit.detail, unit.id)      # Obj_0 不写进 detail
            self.assertNotIn("launcher", unit.detail)
        self.assertTrue(all(g.missile_state(u) == "unassigned"
                            for u in g.units.values() if u.is_loitering_missile()))

    def test_carried_missile_of_destroyed_carrier_is_lost(self):
        g = self.game()
        car, m = g.units["carrier"], g.units["missileA"]
        g.deploy_missile(car.id, m.id)
        car.blood = 0
        g.advance(1)
        self.assertEqual(g.missile_state(m), "lost")


class MissileLaunchGateTests(MissileFixture):
    def test_not_started_is_rejected(self):
        g = Game(seed=3, snapshot=str(self.root))
        car, m = g.units["carrier"], g.units["missileA"]
        g.started = True
        g.deploy_missile(car.id, m.id)
        g.started = False
        ok, why = g.launch_missile(car.id, m.id)
        self.assertFalse(ok)
        self.assertIn("开始前", why)

    def test_carried_blood_zero_can_launch(self):
        g = self.game()
        car, m = g.units["carrier"], g.units["missileA"]
        g.deploy_missile(car.id, m.id)
        self.assertEqual(m.blood, 0)
        ok, why = g.launch_missile(car.id, m.id)
        self.assertTrue(ok, why)
        self.assertEqual(m.blood, m.establishment)
        self.assertEqual(m.pos, car.pos)

    def test_wrong_carrier_and_destroyed_carrier_rejected(self):
        g = self.game()
        car, m, other = g.units["carrier"], g.units["missileA"], g.units["missileB"]
        self.assertIn("4.10(3)", g.launch_missile(car.id, m.id)[1])
        g.deploy_missile(car.id, m.id)
        self.assertIn("4.10(3)", g.launch_missile(car.id, other.id)[1])
        car.blood = 0
        ok, why = g.launch_missile(car.id, m.id)
        self.assertFalse(ok)
        self.assertIn("消灭", why)

    def test_lost_missile_cannot_revive(self):
        g = self.game()
        car, m = g.units["carrier"], g.units["missileA"]
        g.deploy_missile(car.id, m.id)
        g.launch_missile(car.id, m.id)
        g.advance(g.loiter_sec + 1)
        self.assertEqual(g.missile_state(m), "lost")
        self.assertIn("已消耗/已损失", g.launch_missile(car.id, m.id)[1])

    def test_non_vehicle_and_enemy_carrier_rejected(self):
        g = self.game()
        m, squad, foe = g.units["missileA"], g.units["squad"], g.units["foe"]
        self.assertIn("车辆", g.deploy_missile(squad.id, m.id)[1])
        self.assertIn("己方", g.deploy_missile(foe.id, m.id)[1])

    def test_deploy_only_at_opening(self):
        g = self.game()
        car, m = g.units["carrier"], g.units["missileB"]
        g.advance(5)
        ok, why = g.deploy_missile(car.id, m.id)
        self.assertFalse(ok)
        self.assertIn("开始前", why)


class MissileEnvPipelineTests(MissileFixture):
    def test_deploy_and_launch_reach_the_interface(self):
        env = self.env()
        self.assertIn({"type": "load", "unit": "missileA", "carrier": "carrier"},
                      env.legal_actions())
        _obs, _r, _d, info = env.step({"type": "load", "unit": "missileA",
                                       "carrier": "carrier"})
        self.assertFalse(info.get("invalid_action"), info)
        self.assertEqual(env.game.missile_state(env.game.units["missileA"]),
                         "carried")
        self.assertIn({"type": "launch", "unit": "carrier", "missile": "missileA"},
                      env.legal_actions())
        obs, _r, _d, info = env.step({"type": "launch", "unit": "carrier",
                                      "missile": "missileA"})
        self.assertFalse(info.get("invalid_action"), info)
        self.assertEqual(info["seconds"], 10)       # 显式声明的发射耗时（占位值）
        view = next(u for u in obs["own"] if u["id"] == "missileA")
        self.assertEqual(view["missile_state"], "loitering")
        self.assertIsNotNone(view["loiter_seconds_left"])
        carrier = next(u for u in obs["own"] if u["id"] == "carrier")
        self.assertEqual(carrier["missile_in_flight"], "missileA")

    def test_unload_does_not_disembark_missiles(self):
        """下车只针对人员/无人车；巡飞弹留在车上（它只能发射，不能“取回”）。"""
        env = self.env()
        env.step({"type": "load", "unit": "missileA", "carrier": "carrier"})
        env.step({"type": "load", "unit": "squad", "carrier": "carrier"})
        carrier = env.game.units["carrier"]
        self.assertEqual(sorted(carrier.detail["carried"]), ["missileA", "squad"])
        _o, _r, _d, info = env.step({"type": "unload", "unit": "carrier"})
        self.assertFalse(info.get("invalid_action"), info)
        self.assertEqual(carrier.detail["carried"], ["missileA"])
        m = env.game.units["missileA"]
        self.assertEqual(env.game.missile_state(m), "carried")
        self.assertEqual(m.detail.get("carrier"), "carrier")
        self.assertEqual(env.game.units["squad"].pos, carrier.pos)   # 人员正常下车
        # 车上只剩弹 ⇒ 下车动作不可用（不静默成功、也不“把弹取回”）
        self.assertNotIn({"type": "unload", "unit": "carrier"},
                         env.legal_actions())
        info = env.step({"type": "unload", "unit": "carrier"})[3]
        self.assertTrue(info.get("invalid_action"), info)
        self.assertEqual(env.game.missile_state(env.game.units["missileA"]),
                         "carried")

    def test_launch_removes_carried_and_blocks_repeat(self):
        env = self.env()
        env.step({"type": "load", "unit": "missileA", "carrier": "carrier"})
        env.step({"type": "launch", "unit": "carrier", "missile": "missileA"})
        self.assertNotIn("missileA",
                         env.game.units["carrier"].detail.get("carried") or [])
        info = env.step({"type": "launch", "unit": "carrier",
                         "missile": "missileA"})[3]
        self.assertTrue(info.get("invalid_action"), info)

    def test_one_in_flight_per_carrier_then_release(self):
        env = self.env()
        env.step([{"type": "load", "unit": "missileA", "carrier": "carrier"},
                  {"type": "load", "unit": "missileB", "carrier": "carrier"}])
        env.step({"type": "launch", "unit": "carrier", "missile": "missileA"})
        self.assertFalse([a for a in env.legal_actions()
                          if a.get("type") == "launch"
                          and a.get("missile") == "missileB"])
        m_a = env.game.units["missileA"]
        m_a.detail["loiter_until"] = env.game.second + 1
        env.step({"type": "wait", "seconds": 5})
        self.assertEqual(env.game.missile_state(m_a), "lost")
        offered = [a for a in env.legal_actions()
                   if a.get("type") == "launch" and a.get("missile") == "missileB"]
        self.assertTrue(offered)
        env.step(offered[0])
        self.assertEqual(env.game.missile_state(env.game.units["missileB"]),
                         "loitering")
        env.game.units["carrier"].blood = 0
        env.step({"type": "wait", "seconds": 5})
        self.assertEqual(env.game.missile_state(env.game.units["missileB"]), "lost")
        self.assertNotIn("launched_missile", env.game.units["carrier"].detail)

    def test_attack_consumes_the_missile_once(self):
        env = self.env()
        env.step({"type": "load", "unit": "missileA", "carrier": "carrier"})
        env.step({"type": "launch", "unit": "carrier", "missile": "missileA"})
        g = env.game
        m, foe = g.units["missileA"], g.units["foe"]
        self.assertTrue(g.can_observe(m.id, foe.id))
        _o, _r, _d, info = env.step({"type": "attack", "unit": "missileA",
                                     "target": "foe", "weapon": 76})
        self.assertFalse(info.get("invalid_action"), info)
        self.assertEqual(g.missile_state(m), "spent")
        self.assertEqual(m.blood, 0)
        self.assertTrue(m.detail.get("consumed"))
        self.assertNotIn("loiter_until", m.detail)
        self.assertNotIn("target", m.detail)
        self.assertNotIn("launched_missile", g.units["carrier"].detail)
        self.assertTrue(env.step({"type": "attack", "unit": "missileA",
                                  "target": "foe", "weapon": 76})[3]
                        .get("invalid_action"))

    def test_rejected_attack_does_not_consume(self):
        env = self.env()
        env.step({"type": "load", "unit": "missileA", "carrier": "carrier"})
        env.step({"type": "launch", "unit": "carrier", "missile": "missileA"})
        g = env.game
        far = G.make_mapid(10, 14)
        g.units["foe"].pos = far
        g.units["foe"].pos2 = far
        info = env.step({"type": "attack", "unit": "missileA", "target": "foe",
                         "weapon": 76})[3]
        self.assertTrue(info.get("invalid_action"), info)
        m = g.units["missileA"]
        self.assertEqual(g.missile_state(m), "loitering")
        self.assertEqual(m.ammo["a5"], 1)
        self.assertIn("launched_missile", g.units["carrier"].detail)


class MissileTimingTests(MissileFixture):
    def test_room_speed_mapping_is_explicit_and_overrideable(self):
        self.assertEqual(Game.loiter_seconds(3), 400)
        self.assertEqual(Game.loiter_seconds(10), 120)
        self.assertEqual(Game.loiter_seconds(7), 171)          # parseInt 截断
        self.assertEqual(LOITER_SEC_PLACEHOLDER, 120)          # 旧占位值不再被使用
        self.assertNotEqual(Game.loiter_seconds(3), LOITER_SEC_PLACEHOLDER)
        with self.assertRaises(ValueError):
            Game.loiter_seconds(0)
        g = Game(seed=3, snapshot=str(self.root), loiter_sec=77)
        self.assertEqual(g.loiter_sec, 77)
        self.assertIn("override", g.loiter_sec_source)

    def test_deadline_boundary(self):
        g = self.game(loiter_sec=77)
        car, m = g.units["carrier"], g.units["missileA"]
        g.deploy_missile(car.id, m.id)
        g.launch_missile(car.id, m.id)
        deadline = m.detail["loiter_until"]
        self.assertEqual(deadline, 77)
        g.advance(deadline - 1)
        self.assertEqual(g.missile_state(m), "loitering")
        g.advance(1)
        self.assertEqual(g.missile_state(m), "lost")

    def test_state_survives_save_and_restore(self):
        g = self.game()
        car, m = g.units["carrier"], g.units["missileA"]
        g.deploy_missile(car.id, m.id)
        g.launch_missile(car.id, m.id)
        g.advance(10)
        for clone in (copy.deepcopy(g), pickle.loads(pickle.dumps(g))):
            with self.subTest(kind=type(clone).__name__):
                restored = clone.units["missileA"]
                self.assertEqual(clone.missile_state(restored), "loitering")
                self.assertEqual(restored.detail["loiter_until"],
                                 m.detail["loiter_until"])
                clone.advance(clone.loiter_sec)
                self.assertEqual(clone.missile_state(clone.units["missileA"]),
                                 "lost")
                self.assertNotIn("launched_missile",
                                 clone.units["carrier"].detail)


class MissileWrapperTests(MissileFixture):
    def test_realtime_schedule_waits_for_the_deploy_order(self):
        episode = RealtimeEpisode(str(self.root), max_seconds=60, move_radius=1)
        status = episode.reset(seed=3)
        episode.step({"type": "load", "unit": "missileA", "carrier": "carrier"},
                     side=status["current_actor"])
        carrier = next(u for u in episode.observe("BLUE")["own"]
                       if u["id"] == "carrier")
        # 车内记录不下发（平台同款），靠载车的 carrying 才能拿到弹的 id
        self.assertEqual(carrier["carrying"], ["missileA"])
        self.assertFalse([a for a in episode.legal_actions("BLUE")
                          if a.get("type") == "launch"])
        status = episode.status()
        while not status["terminated"] and not (
                status["seconds"] >= 30 and episode.current_actor == "BLUE"):
            status = episode.step({"type": "keep"}, side=episode.current_actor)
        launch = [a for a in episode.legal_actions("BLUE")
                  if a.get("type") == "launch" and a.get("missile") == "missileA"]
        self.assertTrue(launch)
        episode.step(launch[0], side="BLUE")
        self.assertTrue(any(u["id"] == "missileA"
                            and u.get("missile_state") == "loitering"
                            for u in episode.observe("BLUE")["own"]))

    def test_episode_projection_keeps_own_carried_units(self):
        """投影枚举必须带上己方车内单位，否则「发射」动作永远生成不出来。"""
        episode = TwoPlayerEpisode(str(self.root), max_seconds=120, max_steps=40,
                                   move_radius=1)
        episode.reset(seed=3)
        self.assertIn({"type": "load", "unit": "missileA", "carrier": "carrier"},
                      episode.legal_actions("BLUE"))
        episode.step({"type": "load", "unit": "missileA", "carrier": "carrier"},
                     side="BLUE")
        episode.step({"type": "keep"} if {"type": "keep"} in
                     episode.legal_actions("RED") else
                     episode.legal_actions("RED")[0], side="RED")
        self.assertIn({"type": "launch", "unit": "carrier", "missile": "missileA"},
                      episode.legal_actions("BLUE"))


class MissileScenarioTests(unittest.TestCase):
    """真实比赛初设（`scenario/7527`）的初态断言。"""

    def setUp(self):
        from engine import paths
        self.snap = paths.snapshot_dir()

    def test_three_missiles_start_unassigned(self):
        g = Game(seed=1, snapshot=self.snap)
        g.start()
        missiles = [u for u in g.units.values() if u.is_loitering_missile()]
        self.assertEqual(len(missiles), 3)
        self.assertTrue(all(g.missile_state(m) == "unassigned" for m in missiles))
        self.assertTrue(all(not m.detail.get("carrier")
                            and not m.detail.get("launcher") for m in missiles))
        self.assertTrue(all(not u.detail.get("launched_missile")
                            for u in g.units.values()))
        # 弹与它的载车同格，但**没有**任何静默建立的所属关系
        for m in missiles:
            self.assertTrue(any(v.typeid == 2 and v.pos == m.pos
                                for v in g.units.values()))
            self.assertNotIn("car", m.detail)

    def test_scenario_missiles_deploy_then_launch(self):
        g = Game(seed=1, snapshot=self.snap)
        g.start()
        m = next(u for u in sorted(g.units.values(), key=lambda x: str(x.id))
                 if u.is_loitering_missile())
        car = next(u for u in sorted(g.units.values(), key=lambda x: str(x.id))
                   if u.typeid == 2 and u.pos == m.pos and u.color == m.color)
        self.assertTrue(g.deploy_missile(car.id, m.id)[0])
        self.assertTrue(g.launch_missile(car.id, m.id)[0])
        self.assertEqual(g.missile_state(m), "loitering")
        self.assertEqual(m.detail["loiter_seconds"], 400)     # RoomSpeed 3


if __name__ == "__main__":
    unittest.main()
