# -*- coding: utf-8 -*-
"""进度台账与监控的契约：台账只是外挂观测，**不能改变学习结果**。

`tools/monitor_train.py` 是"长跑时能不能看出进展"的唯一依据，所以这里锁三件事：
1. 开了台账后，训练记录/检查点与没开时**逐字节相同**（只写文件，不碰学习）；
2. 台账每局一行、字段自洽（attempted 单调、completed ≤ attempted）；
3. 监控摘要的速率/ETA/近期统计在给定台账上是确定的（纯函数）。
"""
import json
import tempfile
import unittest
from pathlib import Path

from env.match_eval import write_synthetic_snapshot
from env.neural_train import NeuralTrainer, canonical
from tools.monitor_train import format_line, read_rows, summarize

CONFIG = {"policy_config": {"buckets": 32, "hidden_dim": 4},
          "episode_config": {"max_seconds": 20, "max_steps": 100}}


class ProgressLedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.snapshot = write_synthetic_snapshot(self.root / "snapshot")

    def trainer(self, **kwargs):
        return NeuralTrainer(self.snapshot, batch_episodes=2, **CONFIG, **kwargs)

    def test_ledger_does_not_change_learning(self):
        ledger = self.root / "nested" / "progress.jsonl"
        with_ledger = self.trainer(progress_path=ledger)
        without = self.trainer()
        a = with_ledger.train(4)
        b = without.train(4)
        strip = lambda rows: [{k: v for k, v in r.items() if k != "elapsed_seconds"} for r in rows]
        self.assertEqual(strip(a), strip(b))
        self.assertEqual(canonical(with_ledger.checkpoint()), canonical(without.checkpoint()))
        self.assertTrue(ledger.exists())

    def test_ledger_rows_are_consistent(self):
        ledger = self.root / "progress.jsonl"
        trainer = self.trainer(progress_path=ledger)
        trainer.train(4)
        rows = read_rows(ledger)
        self.assertEqual(len(rows), 4)
        attempted = [r["attempted"] for r in rows]
        self.assertEqual(attempted, sorted(attempted))
        self.assertEqual(attempted, list(range(1, 5)))
        for row in rows:
            self.assertLessEqual(row["completed"], row["attempted"])
            self.assertIn(row["side"], ("BLUE", "RED"))
            self.assertIsInstance(row["decisions"], int)
            self.assertGreater(row["wall"], 0)
        # 每 2 局一次更新 ⇒ 最后一行的 updates 是 2
        self.assertEqual(rows[-1]["updates"], 2)

    def test_unwritable_ledger_does_not_break_training(self):
        trainer = self.trainer(progress_path=self.root / "progress.jsonl")
        trainer.progress_path = str(self.root / "missing-dir-that-cannot-exist" / "x" / "y.jsonl")
        # 目录可以创建 ⇒ 正常；这里直接模拟不可写：指向一个目录
        trainer.progress_path = str(self.root)
        records = trainer.train(2)
        self.assertEqual(len(records), 2)
        self.assertIsNone(trainer.progress_path)   # 写不了就自己关掉

    def test_summary_rate_eta_and_format(self):
        rows = [
            {"wall": 1000.0, "attempted": 1, "completed": 1, "updates": 0,
             "own": 300.0, "R": 10.0, "result": "win", "decisions": 20,
             "episode_seconds": 5.0},
            {"wall": 1060.0, "attempted": 2, "completed": 2, "updates": 1,
             "own": 320.0, "R": 20.0, "result": "loss", "decisions": 20,
             "episode_seconds": 5.0},
        ]
        report = summarize(rows, total=4, window=50, now=1060.0)
        self.assertEqual(report["done"], 2)
        self.assertAlmostEqual(report["percent"], 50.0)
        self.assertAlmostEqual(report["rate_per_min"], 1.0)   # 60 s 跑完 1 局 ⇒ 1 局/min
        self.assertAlmostEqual(report["eta_minutes"], 2.0)    # 还剩 2 局
        self.assertEqual(report["recent"]["wins"], 1)
        self.assertEqual(report["recent"]["losses"], 1)
        self.assertAlmostEqual(report["recent"]["mean_R"], 15.0)
        self.assertIn("2/4 局", format_line(report))

    def test_empty_ledger_says_so(self):
        report = summarize([], total=10)
        self.assertEqual(report["done"], 0)
        self.assertIn("台账为空", format_line(report))


if __name__ == "__main__":
    unittest.main()
