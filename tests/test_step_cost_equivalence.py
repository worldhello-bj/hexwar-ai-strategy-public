# -*- coding: utf-8 -*-
"""`step_seconds(..., stack=...)` 的等价契约。

E7 提速把"堆叠检查扫全部单位"换成了**调用方传入的位置快照**（`Game.stack_index()`）。
这个改动必须**完全等价**，所以这里直接对拍两条路径：
- 不传 `stack`（现场数，旧口径）
- 传 `stack`（快照口径）
对合成想定里每个单位 × 每个相邻格逐一比较（秒数与拒绝理由都要一致）。
"""
import tempfile
import unittest
from pathlib import Path

from engine import geometry as G
from engine.game import Game
from env.match_eval import write_synthetic_snapshot


class StackIndexEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.snapshot = write_synthetic_snapshot(Path(self.tmp.name) / "snapshot")
        self.game = Game(seed=7, snapshot=str(self.snapshot))
        self.game.start()
        self.game.second = 0

    def test_index_counts_match_direct_scan(self):
        index = self.game.stack_index()
        direct = {}
        for u in self.game.units.values():
            if u.blood > 0 and u.typeid in (1, 2) and not u.detail.get("carrier"):
                key = (u.color, u.pos)
                direct[key] = direct.get(key, 0) + 1
        self.assertEqual(index, direct)

    def test_stack_argument_matches_direct_scan_for_every_unit(self):
        index = self.game.stack_index()
        checked = 0
        for u in self.game.units.values():
            if not u.alive() or not u.pos:
                continue
            for nb in G.neighbors(u.pos):
                if self.game.hex(nb) is None:
                    continue
                fast = self.game.step_seconds(u, u.pos, nb, stack=index)
                slow = self.game.step_seconds(u, u.pos, nb)
                self.assertEqual(fast, slow, "单位 %s 进入 %s 的两条路径不一致" % (u.id, nb))
                checked += 1
        self.assertGreater(checked, 0, "至少要对拍到一个相邻格")

    def test_ignores_units_already_aboard(self):
        """车内单位不占格位：索引里必须不包含它们（旧口径靠 `carrier` 过滤）。"""
        index = self.game.stack_index()
        for (color, pos), count in index.items():
            direct = [u for u in self.game.units.values()
                      if u.blood > 0 and u.typeid in (1, 2) and not u.detail.get("carrier")
                      and u.color == color and u.pos == pos]
            self.assertEqual(count, len(direct))
        for u in self.game.units.values():
            if u.detail.get("carrier"):
                self.assertEqual(index.get((u.color, u.pos), 0), 0)


if __name__ == "__main__":
    unittest.main()
