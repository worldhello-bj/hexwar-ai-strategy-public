# -*- coding: utf-8 -*-
"""快速枚举（`deep_validate=False`）与参考口径（`deep_validate=True`）的差分用例。

背景：`legal_actions` 原本对**每个候选**做一次整状态事务校验（深拷贝）。真实地图
（2721 格、6 个己方单位、半径 6）会产出 1500+ 个候选，实测 **13 秒/次枚举**，
训练与自博弈都无法接受。现在默认走快速路径：`READONLY_TYPES`（机动/等待/停止）
只做只读判定，其余需要引擎裁决的动作（射击/集火/手雷/夺控/装卸/掩蔽）仍走事务校验。

本用例在合成场景上把两种口径**逐动作集合**对齐，并复查：
  - 枚举是只读的（状态与随机数流不变）；
  - 快速枚举给出的每个候选**确实被 `step()` 接受**；
  - 机动的枚举耗时与 `step` 上报的耗时一致（含冲锋只能走 1 格的手册 4.1.1(4)）。
"""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

from engine import geometry as G
from env.hexwar_env import HexwarEnv


def fingerprint(env):
    g = env.game
    return json.dumps({
        "second": g.second,
        "units": sorted([u.id, u.pos, u.pos2, u.blood, u.keep, u.ste, u.att,
                         json.dumps(u.ammo, sort_keys=True),
                         json.dumps(u.detail, sort_keys=True, default=str)]
                        for u in g.units.values()),
        "flags": [c.get("flag") for c in g.cities],
        "rng": repr(g.rng.getstate()),
        "log": len(g.log),
    }, sort_keys=True, default=str)


class EnumerationEquivalenceTests(unittest.TestCase):
    """合成场景：49 格 + 丛林/禁行 + 夺控点 + 红蓝各步兵/车辆（带武器与弹药）。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.start = G.make_mapid(10, 10)
        nbrs = list(G.neighbors(self.start))
        self.terrain_hex = nbrs[1]
        self.blocked_hex = nbrs[2]
        self.objective = nbrs[3]
        self.enemy_hex = nbrs[4]

        hexes = {str(G.make_mapid(r, c)): {"cond": 0, "elevation": 0, "step": 1}
                 for r in range(7, 14) for c in range(7, 14)}
        hexes[str(self.terrain_hex)] = {"cond": 8, "elevation": 0, "step": 1}   # 丛林地
        hexes[str(self.blocked_hex)] = {"cond": 0, "elevation": 0, "step": 0}   # 禁行区
        units = [
            dict(id="blue_inf", data=0, typeid=1, color="BLUE", blood=2, score=3,
                 pos=self.start, wz="29", a5=5),          # 步兵轻武器（射程 3）
            dict(id="blue_veh", data=0, typeid=2, color="BLUE", blood=4, score=5,
                 pos=self.start, wz="37", a3=5),          # 中型坦克炮
            dict(id="red_inf", data=0, typeid=1, color="RED", blood=2, score=3,
                 pos=self.enemy_hex, wz="29", a5=5),
            dict(id="red_veh", data=0, typeid=2, color="RED", blood=4, score=5,
                 pos=self.enemy_hex, wz="37", a3=5),
        ]
        for name, content in (("map_hexes.json", hexes),
                              ("cities.json", [{"id": "c", "pos": self.objective,
                                                "name": "Synthetic objective",
                                                "flag": "RED"}]),
                              ("units.json", units)):
            (root / name).write_text(json.dumps(content), encoding="utf-8")
        self.root = str(root)
        self.env = HexwarEnv(snapshot=self.root, max_seconds=1800)
        self.env.reset(seed=17)

    # ---- 辅助 ----
    def enumerations(self, radius=2):
        before = fingerprint(self.env)
        fast = self.env.legal_actions(move_radius=radius, deep_validate=False)
        after_fast = fingerprint(self.env)
        deep = self.env.legal_actions(move_radius=radius, deep_validate=True)
        after_deep = fingerprint(self.env)
        self.assertEqual(before, after_fast, "快速枚举改动了状态/随机数流")
        self.assertEqual(before, after_deep, "参考枚举改动了状态/随机数流")
        return fast, deep

    @staticmethod
    def keys(acts):
        return {json.dumps(a, sort_keys=True, default=str) for a in acts}

    def assert_same(self, fast, deep, note=""):
        self.assertEqual(self.keys(fast), self.keys(deep), note)

    # ---- 用例 ----
    def test_fast_matches_reference_across_rollout(self):
        rng = random.Random(2026)
        for step in range(30):
            if self.env.done:
                break
            fast, deep = self.enumerations(radius=2)
            self.assertTrue(fast, "第 %d 步枚举为空" % step)
            self.assert_same(fast, deep, "第 %d 步两种口径不一致" % step)
            self.env.step(rng.choice(fast))

    def test_every_fast_candidate_is_accepted_by_step(self):
        for step in range(6):
            if self.env.done:
                break
            fast, _deep = self.enumerations(radius=2)
            for act in fast:
                probe = copy.deepcopy(self.env)
                _, _, _, info = probe.step(copy.deepcopy(act))
                self.assertFalse(info.get("invalid_action"),
                                 "枚举出的动作被 step 拒绝：%s → %s" % (act, info))
                if act["type"] == "move":
                    self.assertEqual(act["seconds"], info["seconds"],
                                     "机动耗时不一致：%s" % (act,))
            self.env.step(fast[0])

    def test_charge_enumerates_only_single_hop_in_both_modes(self):
        # 手册 4.1.1(4)：每冲锋 200 米转疲劳，1 格 = 200 米 → 一个动作只能走 1 格
        self.env.game.units["blue_inf"].ste = 4
        fast, deep = self.enumerations(radius=3)
        self.assert_same(fast, deep, "冲锋状态下两种口径不一致")
        chasing = [a for a in fast if a["type"] == "move"
                   and a["unit"] == "blue_inf"]
        self.assertTrue(chasing, "冲锋单位应有可执行的 1 格机动")
        self.assertTrue(all(len(a["path"]) == 2 for a in chasing),
                        "冲锋单位被枚举出多格路径：%s" % (chasing[:3],))

    def test_wait_survives_exhausted_clock_in_both_modes(self):
        # `wait` 在参考口径里**豁免**剩余时间上限（超时等待被截断到剩余秒）
        self.env.max_seconds = self.env.game.second + 3
        fast, deep = self.enumerations(radius=2)
        self.assertTrue(any(a["type"] == "wait" for a in fast), "快速口径丢了 wait")
        self.assertTrue(any(a["type"] == "wait" for a in deep), "参考口径丢了 wait")
        self.assert_same(fast, deep, "剩余时间不足时两种口径不一致")
        _, _, _, info = self.env.step({"type": "wait", "seconds": 10})
        self.assertEqual(info["seconds"], 3, "超时等待应被截断到剩余秒")

    def test_move_radius_default_is_configurable(self):
        self.assertEqual(self.env.move_radius, 6)
        self.assertEqual(len(self.env.legal_actions(move_radius=1)),
                         len(self.env.legal_actions(move_radius=1,
                                                    deep_validate=True)))
        narrow = HexwarEnv(snapshot=self.root, move_radius=1)
        narrow.reset(seed=17)
        self.assertEqual(len(narrow.legal_actions()),
                         len(self.env.legal_actions(move_radius=1,
                                                    deep_validate=True)))


if __name__ == "__main__":
    unittest.main()
