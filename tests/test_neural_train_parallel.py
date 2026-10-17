# -*- coding: utf-8 -*-
"""并行 rollout 的契约：**并行度不能改变学习结果**。

采样流只有一种口径：每局按 `(seed, 局种子, 席位)` 独立播种 ⇒ 局与局相互独立、
顺序无关 ⇒ 用进程池并行跑出来的记录与串行**逐字段相同**。
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from env.match_eval import write_synthetic_snapshot
from env.neural_train import NeuralTrainer, canonical

CONFIG = {"policy_config": {"buckets": 32, "hidden_dim": 4},
          "episode_config": {"max_seconds": 20, "max_steps": 100}}


def strip_timing(records):
    """记录里唯一允许不同的字段是墙钟。"""
    return [{k: v for k, v in row.items() if k != "elapsed_seconds"} for row in records]


def strip_eval_timing(report):
    report = copy.deepcopy(report)
    for label in ("initial", "trained"):
        for game in report[label]["games"]:
            game.pop("elapsed_seconds", None)
    return report


def state_only(trainer):
    """检查点里 `rollout_workers` 只是元数据（速度旋钮），其余必须逐字节相同。"""
    data = trainer.checkpoint()
    data.pop("rollout_workers", None)
    return canonical(data)


class ParallelRolloutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.snapshot = write_synthetic_snapshot(Path(self.tmp.name) / "snapshot")

    def trainer(self, **kwargs):
        return NeuralTrainer(self.snapshot, batch_episodes=2, **CONFIG, **kwargs)

    def test_parallel_training_matches_serial_exactly(self):
        """4 局、每 2 局一次更新：串行(1 进程) 与 并行(2 进程) 结果必须完全一致。"""
        serial = self.trainer(rollout_workers=1)
        parallel = self.trainer(rollout_workers=2)

        serial_records = serial.train(4)
        parallel_records = parallel.train(4)

        self.assertEqual(strip_timing(serial_records), strip_timing(parallel_records))
        self.assertEqual(serial.episodes_attempted, parallel.episodes_attempted)
        self.assertEqual(serial.episodes_completed, parallel.episodes_completed)
        self.assertEqual(serial.updates, parallel.updates)
        self.assertEqual(serial.pending, parallel.pending)
        self.assertEqual(serial._pair_buffer, parallel._pair_buffer)
        # 权重、优化器状态、RNG 状态全部一致 ⇒ 检查点除并行度外逐字节相同
        self.assertEqual(state_only(serial), state_only(parallel))

    def test_parallel_evaluation_matches_serial_exactly(self):
        """评测本身顺序无关（动作由派生的 eval_rng 抽、策略冻结），并行只是更快。"""
        serial = self.trainer(rollout_workers=1)
        parallel = self.trainer(rollout_workers=2)
        serial.train(2)
        parallel.train(2)

        expected = serial.evaluate()
        actual = parallel.evaluate()
        self.assertEqual(canonical(strip_eval_timing(expected)),
                         canonical(strip_eval_timing(actual)))
        self.assertEqual([r["seed"] for r in actual["trained"]["games"]],
                         [r["seed"] for r in expected["trained"]["games"]])

    def test_rollout_workers_validation(self):
        for bad in (0, -1, 65, 1.5, True):
            with self.assertRaises(ValueError):
                self.trainer(rollout_workers=bad)

    def test_checkpoint_records_worker_count_but_allows_changing_it(self):
        trainer = self.trainer(rollout_workers=2)
        trainer.train(2)
        path = Path(self.tmp.name) / "ckpt.json"
        trainer.save(path)
        self.assertEqual(json.loads(path.read_text())["rollout_workers"], 2)

        # 并行度可以改（纯速度旋钮）
        reloaded = NeuralTrainer.load(path, self.snapshot, rollout_workers=1)
        self.assertEqual(reloaded.rollout_workers, 1)

    def test_repeated_runs_are_reproducible(self):
        """每局独立播种 ⇒ 同一配置两次训练结果一致（与运行顺序无关）。"""
        first = self.trainer().train(2)
        second = self.trainer().train(2)
        self.assertEqual(strip_timing(first), strip_timing(second))


if __name__ == "__main__":
    unittest.main()
