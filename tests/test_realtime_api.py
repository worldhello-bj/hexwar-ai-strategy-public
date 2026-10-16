# -*- coding: utf-8 -*-
"""实时时间模型与实时决策接口的回归（针对「一局只有十几次决策」的根因）。

覆盖两件事：

1. **引擎时间粒度**：`advance(N)` 必须与 `advance(1)×N` 等价；
   登记路径（手册 4.1.1(6) 解压制后续行）必须**按秒**推进，而不是按调用次数。
2. **实时决策接口**（`env.realtime_api.RealtimeEpisode`）：
   1 Hz、每秒双方各一次决策；持续指令并行推进；`keep` 与 `stop` 区分；
   观测带在执行指令与剩余时间。
"""
import unittest

from engine import geometry as G
from engine import paths
from engine.game import Game
from env.realtime_api import RealtimeEpisode

SNAP = paths.snapshot_dir()


def _open_path(g, unit, want):
    seen = {unit.pos}
    path, cur = [], unit.pos
    for _ in range(want):
        nxt = None
        for nb in G.neighbors(cur):
            if nb in seen:
                continue
            if g.step_seconds(unit, cur, nb)[0] > 0:
                nxt = nb
                break
        if nxt is None:
            break
        seen.add(nxt)
        path.append(nxt)
        cur = nxt
    return path


def _two_units(g, side="BLUE"):
    return [u for u in sorted(g.units.values(), key=lambda x: str(x.id))
            if u.color == side and u.typeid in (1, 2) and u.alive()
            and not u.detail.get("carrier")][:2]


class ClockGranularityTests(unittest.TestCase):
    """`advance` 必须逐秒推进——否则「把动作拆碎」就能多换自动裁决。"""

    def _advance_chunked(self, chunk, window):
        g = Game(snapshot=SNAP)
        g.start()
        g.second = 0
        unit = _two_units(g)[0]
        path = _open_path(g, unit, 8)
        if path:
            g.begin_move(unit.id, path)
        for _ in range(window // chunk):
            g.advance(chunk)
        return g, unit, path

    def test_advance_is_chunk_independent(self):
        one, u1, p1 = self._advance_chunked(1, 60)
        whole, u2, p2 = self._advance_chunked(60, 60)
        self.assertEqual(p1, p2)
        self.assertEqual(u1.pos, u2.pos)
        self.assertEqual(one.pending_moves(), whole.pending_moves())
        self.assertEqual(one.second, whole.second)

    def test_registered_path_is_second_driven(self):
        """60 秒/格的单位，60 秒里只能走 1 格——不管把 60 秒切成几段。"""
        g, unit, path = self._advance_chunked(1, 60)
        self.assertTrue(path)
        rule_seconds = g.step_seconds(unit, path[0], path[1])[0] if len(path) > 1 else 60
        self.assertGreaterEqual(rule_seconds, 60)
        self.assertEqual(unit.pos, path[0], "60 秒/格 ⇒ 60 秒内只应推进 1 格")

    def test_resolution_entry_count_matches_elapsed_seconds(self):
        """自动裁决入口的调用次数 = 经过的秒数（不是调用次数）。"""
        counts = []
        for chunk in (1, 60):
            g = Game(snapshot=SNAP)
            g.start()
            g.second = 0
            seen = []
            original = g.resolve_engagements

            def wrapper(*a, **kw):
                seen.append(1)
                return original(*a, **kw)

            g.resolve_engagements = wrapper
            for _ in range(60 // chunk):
                g.advance(chunk)
            counts.append(len(seen))
        self.assertEqual(counts[0], counts[1])


class RealtimeEpisodeTests(unittest.TestCase):
    def setUp(self):
        self.episode = RealtimeEpisode(SNAP, max_seconds=300, move_radius=1)
        self.status = self.episode.reset(seed=0)

    def _keep_until(self, seconds):
        status = self.status
        while not status["terminated"] and status["seconds"] < seconds:
            status = self.episode.step({"type": "keep"}, side=status["current_actor"])
        self.status = status
        return status

    def test_one_decision_per_side_per_second(self):
        per = {"BLUE": 0, "RED": 0}
        status = self.status
        while not status["terminated"] and status["seconds"] < 20:
            side = status["current_actor"]
            status = self.episode.step({"type": "keep"}, side=side)
            per[side] += 1
        self.assertEqual(per["BLUE"], 20)
        self.assertEqual(per["RED"], 20)

    def test_keep_is_always_legal_and_stop_needs_an_order(self):
        legal = self.episode.legal_actions("BLUE")
        self.assertTrue(any(a["type"] == "keep" for a in legal))
        self.assertFalse(any(a["type"] == "stop" for a in legal),
                         "没有在执行指令时不应出现 stop")
        # 下一条持续指令后，stop 应出现，且与 keep 是两种不同的动作
        moves = [a for a in legal if a["type"] == "move"]
        self.assertTrue(moves)
        status = self.episode.step(moves[0], side="BLUE")
        status = self.episode.step({"type": "keep"}, side="RED")
        legal2 = self.episode.legal_actions("BLUE")
        self.assertTrue(any(a["type"] == "stop" for a in legal2))
        status = self.episode.step({"type": "stop", "unit": moves[0]["unit"]}, side="BLUE")
        self.assertEqual(status["current_actor"], "RED")

    def test_orders_run_in_parallel(self):
        """两个单位相隔一秒拿到指令，两者都要在统一时间轴上推进。"""
        g = self.episode._session.env.game
        a, b = _two_units(g)
        start = {a.id: a.pos, b.id: b.pos}
        legal = self.episode.legal_actions("BLUE")
        order_a = next(a_ for a_ in legal if a_["type"] == "move" and a_["unit"] == a.id)
        self.episode.step(order_a, side="BLUE")
        self.episode.step({"type": "keep"}, side="RED")
        self.episode.step({"type": "keep"}, side="BLUE")
        self.episode.step({"type": "keep"}, side="RED")
        legal_b = self.episode.legal_actions("BLUE")
        order_b = next((a_ for a_ in legal_b if a_["type"] == "move" and a_["unit"] == b.id), None)
        self.assertIsNotNone(order_b, "B 在 t=1 应该能拿到自己的指令")
        self.episode.step(order_b, side="BLUE")
        self.episode.step({"type": "keep"}, side="RED")
        self._keep_until(200)
        self.assertNotEqual(a.pos, start[a.id])
        self.assertNotEqual(b.pos, start[b.id])

    def test_observation_carries_order_progress(self):
        g = self.episode._session.env.game
        unit = _two_units(g)[0]
        legal = self.episode.legal_actions("BLUE")
        order = next(a for a in legal if a["type"] == "move" and a["unit"] == unit.id)
        self.episode.step(order, side="BLUE")
        obs = self.episode.observe("BLUE")
        view = next(v for v in obs["own"] if v["id"] == unit.id)
        self.assertIn("order", view)
        self.assertEqual(view["order"]["kind"], "move")
        self.assertIn("remaining_hexes", view["order"])
        self.assertIn("time_left", obs)

    def test_opening_scores_are_exposed_for_reward_baseline(self):
        self.assertEqual(self.status["opening_scores"], {"BLUE": 234, "RED": 354})
        self.assertIsNone(self.status["final_scores"])

    def test_rejects_unadvertised_action(self):
        with self.assertRaises(ValueError):
            self.episode.step({"type": "move", "unit": "Obj_nonexistent", "path": [1]})

    def test_wrong_seat_is_rejected(self):
        with self.assertRaises(ValueError):
            self.episode.step({"type": "keep"}, side="RED")


if __name__ == "__main__":
    unittest.main()
