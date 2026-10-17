# -*- coding: utf-8 -*-
"""自博弈对手的契约。**对手恒为当前策略自己**，没有 random 选项。

为什么这条测试重要：
- 均匀随机对手太弱且不自洽 —— 练到一定程度后继续与它对练几乎不再提供梯度，
  策略很快撞到天花板。所以对手必须也用策略选招。
- 但对手动作必须可复现，且**不能消耗学习者自己的采样流**（否则同一种子在不同
  并行度下结果会不同）。
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import copy

from env.match_eval import write_synthetic_snapshot
from env.neural_train import NeuralTrainer, _RolloutRunner

CONFIG = {"policy_config": {"buckets": 32, "hidden_dim": 4},
          "episode_config": {"max_seconds": 20, "move_radius": 1}}


class SelfPlayOpponentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.snapshot = write_synthetic_snapshot(Path(self.tmp.name) / "snapshot")

    def trainer(self, **kwargs):
        kwargs.setdefault("interface", "realtime")
        kwargs.setdefault("batch_episodes", 2)
        return NeuralTrainer(self.snapshot, **CONFIG, **kwargs)

    def test_opponent_is_sampled_from_the_policy(self):
        trainer = self.trainer()
        sides = []
        original = trainer.policy.sample

        def sample(observation, candidates, rng=None):
            sides.append(observation.get("side"))
            return original(observation, candidates, rng=rng)

        with patch.object(trainer.policy, "sample", side_effect=sample):
            trainer._rollout(trainer.policy, 11, "BLUE", training=True)
        # 双方都由策略选招 ⇒ 观测里两种席位都会出现
        self.assertEqual(sorted(set(sides)), ["BLUE", "RED"])

    def test_rollout_is_deterministic_per_seed(self):
        first = self.trainer()
        second = self.trainer()
        record_a, _ = first._rollout(first.policy, 11, "RED", training=False)
        record_b, _ = second._rollout(second.policy, 11, "RED", training=False)
        self.assertEqual((record_a["own_score"], record_a["other_score"]),
                         (record_b["own_score"], record_b["other_score"]))

    def test_opponent_does_not_consume_learner_sampling_stream(self):
        """双方各用一条局部流 ⇒ 整局跑完策略自身的 rng 状态一点没动。"""
        trainer = self.trainer()
        before = copy.deepcopy(trainer.policy.rng.bit_generator.state)
        trainer._rollout(trainer.policy, 5, "BLUE", training=True)
        self.assertEqual(before, trainer.policy.rng.bit_generator.state)

    def test_rollout_runner_carries_no_mode_flags(self):
        """并行靠 `_RolloutRunner` 把上下文 pickle 到子进程：旧口径字段必须已删除。"""
        trainer = self.trainer()
        runner = trainer._runner_context()
        self.assertIsInstance(runner, _RolloutRunner)
        for removed in ("reward_mode", "seat_pairing", "policy_seed_mode", "opponent_mode"):
            self.assertFalse(hasattr(runner, removed), removed)


if __name__ == "__main__":
    unittest.main()
