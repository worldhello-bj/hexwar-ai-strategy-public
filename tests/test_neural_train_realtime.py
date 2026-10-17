# -*- coding: utf-8 -*-
"""`interface="realtime"`（1 Hz）训练路径：每方每秒一次决策，机动并行推进。

要点（相对回合制 `TwoPlayerEpisode` 的差别）：
* 一局的决策数 ≈ `max_seconds`（每秒一次），而不是十几条指令；
* 学习目标改成 **终局本方分 − 本方开局分**（席位常数不再直接变成优势）；
* 检查点把 `interface` 记下来，续训时不一致直接拒绝。
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from env.match_eval import write_synthetic_snapshot
from env.neural_train import INTERFACES, NeuralTrainer, realtime_episode_config


class RealtimeTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.snapshot = write_synthetic_snapshot(self.root / "snapshot")

    def trainer(self, **kwargs):
        return NeuralTrainer(self.snapshot, policy_config={"buckets": 32, "hidden_dim": 4},
                             episode_config={"max_seconds": 6, "max_steps": 100},
                             interface="realtime", batch_episodes=2, **kwargs)

    def test_interface_registry_and_config_filter(self):
        """实时接口只吃它认识的键（`max_steps` 这类必须被丢掉）。"""
        self.assertEqual(INTERFACES, ("serial", "realtime"))
        self.assertEqual(realtime_episode_config({"max_seconds": 6, "max_steps": 100,
                                                  "deploy_window": "opening"}),
                         {"max_seconds": 6, "deploy_window": "opening"})

    def test_rollout_has_one_decision_per_second(self):
        """1 Hz：一局 6 秒 ⇒ 学习方最多 6 次决策，且时钟确实在走。"""
        trainer = self.trainer()
        record, trajectory = trainer._rollout(trainer.policy, 42, "BLUE", training=True)
        self.assertLessEqual(record["decisions"], 6)
        self.assertEqual(len(trajectory), record["decisions"])
        self.assertGreaterEqual(record["status"]["seconds"], 1)
        self.assertEqual(record["interface"], "realtime")
        # 学习目标 = 净胜分增量（两边各扣开局分后相减）
        self.assertIsNotNone(record["opening_own"])
        self.assertAlmostEqual(
            record["learning_return"],
            (record["own_score"] - record["opening_own"])
            - (record["other_score"] - record["opening_other"]), places=6)

    def test_train_and_resume_bind_interface(self):
        """训练一轮后能存能续；把 interface 改成别的会被拒。"""
        trainer = self.trainer()
        trainer.train(1)
        path = self.root / "rt.json"
        trainer.save(path)
        data = json.loads(path.read_text())
        self.assertEqual(data["interface"], "realtime")
        resumed = NeuralTrainer.load(path, self.snapshot)
        self.assertEqual(resumed.interface, "realtime")
        self.assertEqual(resumed.episodes_attempted, 1)
        resumed.train(1)
        self.assertEqual(resumed.episodes_attempted, 2)
        # 续训时显式要求另一种接口 → 拒绝
        other = self.root / "other.json"
        data["interface"] = "serial"
        other.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            NeuralTrainer.load(other, self.snapshot, interface="realtime")
        self.assertEqual(NeuralTrainer.load(other, self.snapshot).interface, "serial")

    def test_realtime_rejects_schema_two(self):
        with self.assertRaises(ValueError):
            NeuralTrainer(self.snapshot, interface="realtime", observation_schema=2,
                          policy_config={"buckets": 32, "hidden_dim": 4})
        with self.assertRaises(ValueError):
            NeuralTrainer(self.snapshot, interface="nonsense",
                          policy_config={"buckets": 32, "hidden_dim": 4})

    def test_serial_default_is_unchanged(self):
        """不传 interface 时仍是回合制，且 `learning_return` 就是终局本方分。"""
        trainer = NeuralTrainer(self.snapshot, policy_config={"buckets": 32, "hidden_dim": 4},
                                episode_config={"max_seconds": 6, "max_steps": 100})
        self.assertEqual(trainer.interface, "serial")
        record, _ = trainer._rollout(trainer.policy, 7, "RED", training=False)
        self.assertEqual(record["interface"], "serial")
        self.assertEqual(record["learning_return"], record["own_score"])


if __name__ == "__main__":
    unittest.main()
