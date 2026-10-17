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
from unittest.mock import patch

from engine import geometry as G
from engine.game import Game
from env.episode_api import TwoPlayerEpisode
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


class VisibilitySnapshotTests(unittest.TestCase):
    """`_can_observe` 快照的契约：与直接调用一致 + 状态一变就失效 + 被替换时完全不走缓存。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        snapshot = write_synthetic_snapshot(Path(self.tmp.name) / "snapshot")
        self.episode = TwoPlayerEpisode(str(snapshot), public_score=False)
        self.episode.reset(seed=11)
        self.env = self.episode._session.env
        self.game = self.env.game

    def _pairs(self):
        own = [u for u in self.game.units.values() if u.alive() and u.color == "BLUE"]
        other = [u for u in self.game.units.values() if u.alive()]
        return [(o.id, t.id) for o in own for t in other]

    def test_snapshot_matches_direct_call(self):
        self.env._sync_visibility()
        for obs_id, tgt_id in self._pairs():
            self.assertEqual(self.env._can_observe(obs_id, tgt_id),
                             self.game.can_observe(obs_id, tgt_id))

    def test_state_change_invalidates_snapshot(self):
        self.env._sync_visibility()
        pairs = self._pairs()
        self.assertTrue(pairs)
        self.env._can_observe(*pairs[0])
        self.assertGreater(len(self.env._vis), 0)
        victim = next(u for u in self.game.units.values() if u.color == "RED")
        victim.blood += 1                      # 指纹变了
        self.env._sync_visibility()
        self.assertEqual(self.env._vis, {})

    def test_monkeypatched_can_observe_bypasses_cache(self):
        """遮雾测试会把 `can_observe` patch 成 False；那时指纹不变但实现变了，
        必须完全不缓存，否则会拿重置时缓存的 True 当结果（已踩过）。"""
        with patch.object(self.game, "can_observe", return_value=False):
            before = self.episode.observe("BLUE")
            victim = next(u for u in self.game.units.values() if u.color == "RED")
            victim.blood += 5
            victim.score += 100
            self.assertEqual(self.episode.observe("BLUE"), before)   # 变血/变分不得泄露


if __name__ == "__main__":
    unittest.main()