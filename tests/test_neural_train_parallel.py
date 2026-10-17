# -*- coding: utf-8 -*-
"""并行 rollout 的契约：**并行度不能改变学习结果**。

核心不变式：`rollout_workers` 只是速度旋钮。
- `policy_seed_mode="per_episode"` 下每局按 `(seed, 局种子, 席位)` 重新播种
  ⇒ 局与局相互独立、顺序无关 ⇒ 用进程池并行跑出来的记录与串行**逐字段相同**。
- `policy_seed_mode="shared"`（历史默认）是一条跨局沿用的随机流，**不能并行**，
  所以 `rollout_workers>1` 时必须显式走 per_episode。
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from env.match_eval import write_synthetic_snapshot
from env.neural_train import NeuralTrainer, canonical, main

CONFIG = {"policy_config": {"buckets": 32, "hidden_dim": 4},
          "episode_config": {"max_seconds": 20, "max_steps": 100}}


def strip_timing(records):
    """记录里唯一允许不同的字段是墙钟与熵（熵是描述量，但并行不改变采样也不改变它）。"""
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
        serial = self.trainer(rollout_workers=1, policy_seed_mode="per_episode")
        parallel = self.trainer(rollout_workers=2)
        self.assertEqual(parallel.policy_seed_mode, "per_episode")

        serial_records = serial.train(4)
        parallel_records = parallel.train(4)

        self.assertEqual(strip_timing(serial_records), strip_timing(parallel_records))
        self.assertEqual(serial.episodes_attempted, parallel.episodes_attempted)
        self.assertEqual(serial.episodes_completed, parallel.episodes_completed)
        self.assertEqual(serial.updates, parallel.updates)
        self.assertEqual(serial.pending, parallel.pending)
        # 权重、优化器状态、RNG 状态全部一致 ⇒ 检查点除并行度外逐字节相同
        self.assertEqual(state_only(serial), state_only(parallel))

    def test_parallel_evaluation_matches_serial_exactly(self):
        """评测本身顺序无关（动作由派生的 eval_rng 抽、策略冻结），并行只是更快。"""
        serial = self.trainer(rollout_workers=1, policy_seed_mode="per_episode")
        parallel = self.trainer(rollout_workers=2)
        serial.train(2)
        parallel.train(2)

        expected = serial.evaluate()
        actual = parallel.evaluate()
        self.assertEqual(canonical(strip_eval_timing(expected)),
                         canonical(strip_eval_timing(actual)))
        self.assertEqual([r["seed"] for r in actual["trained"]["games"]],
                         [r["seed"] for r in expected["trained"]["games"]])

    def test_shared_mode_still_defaults_for_serial(self):
        trainer = self.trainer()
        self.assertEqual(trainer.rollout_workers, 1)
        self.assertEqual(trainer.policy_seed_mode, "shared")

    def test_workers_above_one_require_per_episode_mode(self):
        with self.assertRaises(ValueError):
            self.trainer(rollout_workers=2, policy_seed_mode="shared")

    def test_rollout_workers_validation(self):
        for bad in (0, -1, 65, 1.5, True):
            with self.assertRaises(ValueError):
                self.trainer(rollout_workers=bad)
        with self.assertRaises(ValueError):
            self.trainer(policy_seed_mode="bogus")

    def test_checkpoint_binds_policy_seed_mode_but_not_worker_count(self):
        trainer = self.trainer(rollout_workers=2)
        trainer.train(2)
        path = Path(self.tmp.name) / "ckpt.json"
        trainer.save(path)
        data = json.loads(path.read_text())
        self.assertEqual(data["rollout_workers"], 2)
        self.assertEqual(data["policy_seed_mode"], "per_episode")

        # 并行度可以改（纯速度旋钮），采样口径跟着检查点走
        reloaded = NeuralTrainer.load(path, self.snapshot, rollout_workers=1)
        self.assertEqual(reloaded.rollout_workers, 1)
        self.assertEqual(reloaded.policy_seed_mode, "per_episode")

    def test_cli_rejects_policy_seed_mode_change_on_resume(self):
        """改了采样口径就不是同一条随机实现，续跑必须报错而不是静默继续。"""
        trainer = self.trainer(rollout_workers=2)
        trainer.train(2)
        path = Path(self.tmp.name) / "ckpt.json"
        trainer.save(path)
        with self.assertRaises(SystemExit):
            main(["--snapshot", str(self.snapshot), "--resume", str(path),
                  "--policy-seed-mode", "shared",
                  "--checkpoint", str(Path(self.tmp.name) / "new.json"),
                  "--report", str(Path(self.tmp.name) / "new.json.report")])

    def test_serial_shared_mode_is_unchanged_by_the_refactor(self):
        """历史口径（shared）必须仍然可复现：同一配置两次训练结果一致。"""
        first = self.trainer().train(2)
        second = self.trainer().train(2)
        self.assertEqual(strip_timing(first), strip_timing(second))


if __name__ == "__main__":
    unittest.main()
