"""Environment API regression tests using small, original synthetic fixtures."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

from engine import geometry as G
from env.hexwar_env import HexwarEnv


class EnvContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.start = G.make_mapid(10, 10)
        self.neighbor = G.neighbors(self.start)[0]
        self.remote = G.make_mapid(35, 35)
        hexes = {str(G.make_mapid(r, c)): {"cond": 0, "elevation": 0, "step": 1}
                 for r in range(7, 14) for c in range(7, 14)}
        hexes[str(self.remote)] = {"cond": 0, "elevation": 0, "step": 1}
        units = [dict(id="a", data=0, typeid=1, color="BLUE", blood=2,
                      score=3, pos=self.start),
                 dict(id="b", data=0, typeid=1, color="RED", blood=2,
                      score=4, pos=self.remote)]
        for filename, content in (("map_hexes.json", hexes),
                                  ("cities.json", [{"id": "c", "pos": self.neighbor,
                                                    "name": "Synthetic objective", "flag": "RED"}]),
                                  ("units.json", units)):
            (self.root / filename).write_text(json.dumps(content))
        self.env = HexwarEnv(snapshot=str(self.root))
        self.env.reset(seed=17)

    def state(self):
        g = self.env.game
        return copy.deepcopy({"obs": self.env.observe(), "history": self.env._history,
                              "log": g.log, "orders": g.orders,
                              "units": {uid: {slot: getattr(u, slot) for slot in u.__slots__}
                                        for uid, u in g.units.items()},
                              "rng": g.rng.getstate()})

    def assert_invalid(self, action):
        before = self.state()
        original = self.env.game
        _, reward, _, info = self.env.step(action)
        self.assertTrue(info.get("invalid_action"), info)
        self.assertEqual(reward, 0)
        self.assertIs(self.env.game, original)
        self.assertEqual(self.state(), before)

    def test_fog_does_not_disclose_never_seen_units(self):
        obs = self.env.observe()
        self.assertEqual(obs["enemy_visible"], [])
        self.assertEqual(obs["enemy_known"], [])
        self.env.game.units["b"].pos = self.neighbor
        self.env._remember()
        self.env.game.units["b"].pos = self.remote
        known = self.env.observe()["enemy_known"]
        self.assertEqual(known[0]["pos"], self.neighbor)
        self.assertEqual(known[0]["last_seen"], 0)

    def test_observations_are_detached(self):
        self.env.game.units["a"].detail["carried"] = ["synthetic-passenger"]
        before = self.state()
        obs = self.env.observe()
        obs["own"][0]["carrying"].append("injected")
        obs["own"][0]["blood"] = -1
        obs["objectives"][0]["held_by"] = "BLUE"
        obs["score"]["own"]["total"] = -999
        self.assertEqual(self.state(), before)
        json.dumps(obs)

    def test_wrong_side_dead_loaded_and_stunned_units_rejected(self):
        self.assert_invalid({"type": "hide", "unit": "b", "on": 1})
        for attr, value in (("blood", 0), ("stunned_until", 100)):
            u = self.env.game.units["a"]
            old = getattr(u, attr)
            setattr(u, attr, value)
            self.assert_invalid({"type": "hide", "unit": "a", "on": 1})
            setattr(u, attr, old)
        self.env.game.units["a"].detail["carrier"] = "car"
        self.assert_invalid({"type": "hide", "unit": "a", "on": 1})

    def test_malformed_inputs_do_not_mutate(self):
        for action in (None, 3, "wait", {}, {"type": "nonsense"},
                       {"type": "wait", "seconds": -1},
                       {"type": "wait", "seconds": 0},
                       {"type": "wait", "seconds": "10"},
                       {"type": "wait", "seconds": True},
                       {"type": "move", "unit": "a", "path": [self.remote]},
                       {"type": "attack", "unit": "a", "target": "a"}):
            with self.subTest(action=action):
                self.assert_invalid(action)

    def test_invalid_move_and_batch_are_atomic(self):
        missing = next(h for h in G.neighbors(self.neighbor) if h != self.start)
        self.env.game.hexes.pop(missing, None)
        self.assert_invalid({"type": "move", "unit": "a",
                             "path": [self.start, self.neighbor, missing]})
        self.assert_invalid([{"type": "hide", "unit": "a", "on": 1},
                             {"type": "hide", "unit": "b", "on": 1}])

    def test_batch_repeated_unit_is_rejected(self):
        self.assert_invalid([{"type": "hide", "unit": "a", "on": 1},
                             {"type": "hide", "unit": "a", "on": 0}])

    def test_success_preserves_shared_game_identity(self):
        shared = self.env.game
        unit = shared.units["a"]
        self.env.step({"type": "wait", "seconds": 4})
        self.assertIs(self.env.game, shared)
        self.assertEqual(shared.second, 4)
        self.assertIs(unit, shared.units["a"])

    def test_seeded_stochastic_steps_replay_exactly(self):
        def configure():
            self.env.reset(seed=36)
            self.env.game.units["a"].wz = [144]
            self.env.game.units["a"].ammo["a5"] = 5
            self.env.game.units["b"].pos = self.neighbor
            self.env.game.units["b"].blood = 10
        action = {"type": "grenade", "unit": "a", "target_hex": self.neighbor}
        configure()
        first = [self.env.step(action) for _ in range(3)]
        configure()
        second = []
        for _ in range(3):
            random.seed(999)
            self.env.legal_actions(move_radius=1)
            second.append(self.env.step(action))
        self.assertEqual(first, second)
        self.assertTrue(all(not result[3].get("invalid_action") for result in first))

    def test_empty_ammo_attacks_are_not_enumerated(self):
        self.env.game.units["a"].wz = [29]
        self.env.game.units["b"].pos = self.neighbor
        acts = self.env.legal_actions(move_radius=0)
        self.assertFalse(any(a["type"] in ("attack", "area_fire") for a in acts))
        self.assert_invalid({"type": "attack", "unit": "a", "target": "b",
                             "weapon": 29, "track": 1})

    def test_initial_elimination_is_terminal(self):
        units = json.loads((self.root / "units.json").read_text())
        units[1]["blood"] = 0
        (self.root / "units.json").write_text(json.dumps(units))
        self.assertTrue(self.env.reset(seed=1)["done"])

    def test_wait_inside_batch_advances_time(self):
        _, _, _, info = self.env.step([{"type": "wait", "seconds": 7},
                                       {"type": "wait", "seconds": 12}])
        self.assertEqual(self.env.game.second, 12)
        self.assertEqual(info["seconds"], 12)

    def test_reset_initial_terminal_and_terminal_noop(self):
        self.env.max_seconds = 0
        self.assertTrue(self.env.reset(seed=17)["done"])
        self.assertEqual(self.env.legal_actions(), [])
        before = self.state()
        _, reward, done, _ = self.env.step({"type": "wait"})
        self.assertTrue(done)
        self.assertEqual(reward, 0)
        self.assertEqual(before, self.state())

    def test_time_limit_does_not_allow_free_long_moves(self):
        self.env.max_seconds = 5
        self.assert_invalid({"type": "move", "unit": "a", "path": [self.neighbor]})
        _, _, done, info = self.env.step({"type": "wait", "seconds": 10})
        self.assertEqual(info["seconds"], 5)
        self.assertTrue(done)
        self.assertEqual(self.env.game.second, 5)

    def test_reward_equals_score_difference_delta(self):
        before = self.env.score()["diff"]
        obs, reward, _, info = self.env.step({"type": "move", "unit": "a",
                                            "path": [self.start, self.neighbor]})
        self.assertFalse(info.get("invalid_action"), info)
        self.assertEqual(reward, obs["score"]["diff"] - before)
        # 夺占 1 个夺控点：夺控分与歼灭分现在都**已实测**（进奖励），但这个合成 fixture 的
        # 夺控点名称不在实测表里（主要 80 / 次要 50）⇒ **不得分**，并且要显式报出
        # “名称没测过”——不猜、不静默当 10 分。所以这局走到相邻格后 reward 仍为 0。
        # 见 `docs/计分规则待确认.md`。
        self.assertEqual(reward, 0)
        self.assertEqual(self.env.score()["pending_rules"], [])   # 没有悬而未决的规则
        self.assertIn("establishment_from_load_blood", self.env.score()["unverified"])
        # 合成 fixture 的夺控点名称不在实测表里（主要 80 / 次要 50）⇒ **两种口径都不得分**，
        # 并且要显式报出“名称没测过”——不猜、不静默当 10 分。
        platform = HexwarEnv(snapshot=str(self.root), scoring_mode="platform")
        platform.reset(seed=17)
        _, platform_reward, _, _ = platform.step(
            {"type": "move", "unit": "a", "path": [self.start, self.neighbor]})
        self.assertEqual(platform_reward, 0)
        self.assertIn("capture_point_name_unknown", platform.score()["unverified"])
        self.assertEqual(platform.score()["mode"], "platform")

    def test_enumeration_is_readonly_and_every_action_is_accepted(self):
        before = self.state()
        acts = self.env.legal_actions(move_radius=2)
        self.assertEqual(self.state(), before)
        initial = copy.deepcopy(self.env)
        self.assertTrue(acts)
        for act in acts:
            self.env = copy.deepcopy(initial)
            _, _, _, info = self.env.step(act)
            self.assertFalse(info.get("invalid_action"), (act, info))
            if act["type"] == "move":
                self.assertEqual(act["seconds"], info["seconds"])

    def test_charging_enumeration_is_readonly(self):
        self.env.game.units["a"].ste = 4
        before = self.state()
        self.env.legal_actions(move_radius=2)
        self.assertEqual(self.state(), before)

    def test_seeded_random_stream_is_isolated(self):
        other = HexwarEnv(snapshot=str(self.root))
        other.reset(seed=17)
        expected = [other.game.rng.randint(1, 6) for _ in range(12)]
        actual = []
        for _ in range(12):
            random.random()
            other.reset(seed=999)
            actual.append(self.env.game.rng.randint(1, 6))
        self.assertEqual(expected, actual)
        self.env.reset(seed=17)
        self.assertEqual(expected, [self.env.game.rng.randint(1, 6) for _ in range(12)])


if __name__ == "__main__":
    unittest.main()
