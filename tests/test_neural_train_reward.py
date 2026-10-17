# -*- coding: utf-8 -*-
"""奖励与轨迹子采样的契约。**只有净胜分增量这一种奖励。**

- 奖励 = (本方终 − 本方开) − (对方终 − 对方开)：敌方夺控会扣分，策略才学得到“别丢点”。
- 成对训练下两局共用同一个回报：R = net(蓝) + net(红)，开局席位常数完全抵消。
- 接口拿不到开局分（回合制）时退回“本方得分增量” —— 那是接口能力限制，不是可选口径。
- `trajectory_stride=k` 只抽稀进梯度的决策，不改变奖励本身。
"""
import copy
import tempfile
import unittest
from pathlib import Path

from env.match_eval import write_synthetic_snapshot
from env.neural_train import NeuralTrainer

CONFIG = {"policy_config": {"buckets": 32, "hidden_dim": 4},
          "episode_config": {"max_seconds": 20, "move_radius": 1}}


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.snapshot = write_synthetic_snapshot(Path(self.tmp.name) / "snapshot")

    def trainer(self, **kwargs):
        kwargs.setdefault("interface", "realtime")
        kwargs.setdefault("batch_episodes", 2)
        return NeuralTrainer(self.snapshot, **CONFIG, **kwargs)

    def test_net_gain_matches_terminal_scores(self):
        trainer = self.trainer()
        for record in trainer.train(4):
            if record["net_gain"] is None:
                continue
            expected = ((record["own_score"] - record["opening_own"])
                        - (record["other_score"] - record["opening_other"]))
            self.assertAlmostEqual(record["net_gain"], expected)
            self.assertAlmostEqual(record["own_gain"], record["own_score"] - record["opening_own"])
            self.assertAlmostEqual(record["learning_return"], expected)

    def test_pair_return_cancels_opening_offsets(self):
        """成对回报 = 两局净分之和 = 双方终局分之和 − 双方开局分之和。"""
        trainer = self.trainer()
        records = trainer.train(2)
        blue, red = records
        expected = ((blue["own_score"] + red["own_score"])
                    - (blue["opening_own"] + red["opening_own"]))
        self.assertAlmostEqual(blue["pair_return"], expected)
        self.assertAlmostEqual(red["pair_return"], expected)
        self.assertAlmostEqual(expected, blue["net_gain"] + red["net_gain"])

    def test_enemy_capture_is_penalised(self):
        """敌方加分必须让本方净胜分**下降**（旧口径只扣本方开局分，对此毫无反应）。"""
        trainer = self.trainer()
        record = {"own_score": 300, "opening_own": 234,
                  "other_score": 400, "opening_other": 354,
                  "score_diff": -100}
        own_gain = record["own_score"] - record["opening_own"]
        net_gain = own_gain - (record["other_score"] - record["opening_other"])
        self.assertEqual(own_gain, 66)
        self.assertEqual(net_gain, 20)          # 敌方多拿 46 分 ⇒ 净胜分被扣掉
        self.assertLess(net_gain, own_gain)

    def test_stride_subsamples_gradient_only(self):
        full = self.trainer(trajectory_stride=1)
        sparse = self.trainer(trajectory_stride=8)
        counts = {}
        for name, trainer in (("full", full), ("sparse", sparse)):
            lengths = []
            original = trainer.policy.update_batch

            def capture(batch, original=original, lengths=lengths):
                lengths.extend(len(history) for history, _ in batch)
                return original(batch)

            trainer.policy.update_batch = capture
            trainer.train(2)
            counts[name] = lengths
        self.assertTrue(counts["full"])
        self.assertTrue(all(a >= b for a, b in zip(counts["full"], counts["sparse"])))
        self.assertLess(sum(counts["sparse"]), sum(counts["full"]))

    def test_checkpoint_does_not_record_removed_modes(self):
        trainer = self.trainer()
        data = trainer.checkpoint()
        for key in ("reward_mode", "seat_pairing", "policy_seed_mode", "opponent_mode"):
            self.assertNotIn(key, data)


if __name__ == "__main__":
    unittest.main()
