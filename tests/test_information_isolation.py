"""Information-boundary regressions with original, unarmed synthetic pieces.

Public aggregate scores are part of the existing observation contract. These
checks do not promise secrecy for score changes or change any game rules.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from engine import geometry as G
from env.hexwar_env import HexwarEnv


class InformationIsolationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.blue = G.make_mapid(10, 10)
        self.red = G.make_mapid(35, 35)
        self.remote_blue = G.make_mapid(45, 45)
        self.remote_red = G.make_mapid(55, 55)
        hexes = {str(G.make_mapid(r, c)):
                 {"cond": 0, "elevation": 0, "step": 1}
                 for origin in (10, 35, 45, 55)
                 for r in range(origin - 2, origin + 3)
                 for c in range(origin - 2, origin + 3)}
        units = [dict(id="blue", name="Synthetic blue piece", data=0,
                      typeid=1, color="BLUE", blood=2, score=3, pos=self.blue),
                 dict(id="red", name="Synthetic red piece", data=0,
                      typeid=1, color="RED", blood=2, score=4, pos=self.red)]
        for name, value in (("map_hexes.json", hexes),
                            ("units.json", units),
                            ("cities.json", [{"id": "objective", "name": "Synthetic",
                                              "pos": self.blue, "flag": "BLUE"}])):
            (self.root / name).write_text(json.dumps(value), encoding="utf-8")

    def make_env(self, side="BLUE", seed=17):
        env = HexwarEnv(snapshot=str(self.root), side=side, fog=True)
        env.reset(seed=seed)
        self.assertTrue(all(not unit.wz for unit in env.game.units.values()))
        return env

    @staticmethod
    def state(env):
        """Compare all mutable state, including private history and random state."""
        game = dict(env.game.__dict__)
        game["rng"] = env.game.rng.getstate()
        game["units"] = {uid: {slot: getattr(unit, slot) for slot in unit.__slots__}
                         for uid, unit in env.game.units.items()}
        # Wall-clock initialization is independent of the simulated state.
        game.pop("t0", None)
        game.pop("start_str", None)
        return copy.deepcopy({"game": game, "history": env._history,
                              "done": env.done, "observation": env.observe()})

    def test_never_observed_position_changes_do_not_change_observation(self):
        for side, hidden_id, destination in (("BLUE", "red", self.remote_red),
                                             ("RED", "blue", self.remote_blue)):
            with self.subTest(side=side):
                env = self.make_env(side)
                before = env.observe()
                self.assertEqual(before["enemy_visible"], [])
                self.assertEqual(before["enemy_known"], [])
                actions = env.legal_actions(move_radius=1)
                env.game.units[hidden_id].pos = destination
                env._remember()
                self.assertEqual(env.observe(), before)
                self.assertEqual(env.legal_actions(move_radius=1), actions)

    def test_hidden_position_changes_preserve_last_observation(self):
        env = self.make_env()
        hidden = env.game.units["red"]
        last_position = G.neighbors(self.blue)[0]
        hidden.pos = last_position
        env._remember()
        self.assertEqual(env.observe()["enemy_visible"][0]["pos"], last_position)
        hidden.pos = self.red
        env._remember()
        before = env.observe()
        self.assertEqual(before["enemy_visible"], [])
        self.assertEqual(before["enemy_known"][0]["pos"], last_position)
        hidden.pos = self.remote_red
        env._remember()
        self.assertEqual(env.observe(), before)

    def test_public_score_is_explicitly_outside_hidden_detail_guarantee(self):
        env = self.make_env()
        before = env.observe()
        env.game.units["red"].score += 1
        after = env.observe()
        self.assertNotEqual(before["score"], after["score"])
        before.pop("score")
        after.pop("score")
        self.assertEqual(before, after)

    @staticmethod
    def corrupt_containers(value):
        """Mutate nested containers before overwriting the outer container."""
        if isinstance(value, dict):
            for child in list(value.values()):
                InformationIsolationTests.corrupt_containers(child)
            for key in list(value):
                value[key] = "injected"
            value["injected"] = ["injected"]
        elif isinstance(value, list):
            for child in list(value):
                InformationIsolationTests.corrupt_containers(child)
            value[:] = ["injected"]

    def test_returned_observations_cannot_mutate_internal_state(self):
        env = self.make_env()
        # Populate both nested own data and a legitimate last-seen record.
        env.game.units["blue"].detail["carried"] = ["synthetic-passenger"]
        env.game.units["red"].pos = G.neighbors(self.blue)[0]
        env._remember()
        for position in (G.neighbors(self.blue)[0], self.red):
            with self.subTest(position=position):
                env.game.units["red"].pos = position
                before = self.state(env)
                observation = env.observe()
                json.dumps(observation)
                self.corrupt_containers(observation)
                self.assertEqual(self.state(env), before)

    def test_returned_legal_actions_cannot_mutate_internal_state(self):
        env = self.make_env()
        before = self.state(env)
        actions = env.legal_actions(move_radius=1)
        self.assertEqual(self.state(env), before)
        self.assertTrue(any(action["type"] == "move" and action["path"]
                            for action in actions))
        self.assertFalse(any(action["type"] in {"attack", "grenade", "area_fire"}
                             for action in actions))
        expected = copy.deepcopy(actions)
        self.corrupt_containers(actions)
        self.assertEqual(self.state(env), before)
        self.assertEqual(env.legal_actions(move_radius=1), expected)

    def test_interleaved_environments_do_not_contaminate_each_other(self):
        left, right = self.make_env(), self.make_env(side="RED", seed=29)
        left_control = self.make_env()
        right_control = self.make_env(side="RED", seed=29)
        for seconds in (1, 3, 7):
            right_before = self.state(right)
            left.legal_actions(move_radius=1)
            left.step({"type": "wait", "seconds": seconds})
            left_control.step({"type": "wait", "seconds": seconds})
            self.assertEqual(self.state(right), right_before)
            self.assertEqual(self.state(left), self.state(left_control))
            left_before = self.state(left)
            right.legal_actions(move_radius=1)
            right.step({"type": "wait", "seconds": seconds + 1})
            right_control.step({"type": "wait", "seconds": seconds + 1})
            self.assertEqual(self.state(left), left_before)
            self.assertEqual(self.state(right), self.state(right_control))
        left_before = self.state(left)
        right.reset(seed=999)
        right.game.units["blue"].pos = self.red
        right._remember()
        right.game.hexes[self.blue]["elevation"] = 99
        self.assertEqual(self.state(left), left_before)
        self.assertEqual(self.state(left), self.state(left_control))


if __name__ == "__main__":
    unittest.main()
