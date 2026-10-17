# -*- coding: utf-8 -*-
"""学习目标的口径契约（`REWARD_MODES`）与轨迹子采样。

核心两条：
1. `net_gain` = 净胜分增量 = (own终 − own开) − (opp终 − opp开)。
   旧口径 `own_gain` 只看本方分，**敌方夺控拉分时毫无反应** ⇒ 学不到“别丢点”。
2. `trajectory_stride` 只影响**进梯度的条数**，不能改变一局的结果/决策，
   也不改变学习目标的数值。
"""
import json
import tempfile
import unittest
from pathlib import Path

from env.match_eval import write_synthetic_snapshot
from env.neural_train import NeuralTrainer, main

CONFIG = {"policy_config": {"buckets": 32, "hidden_dim": 4},
          "episode_config": {"max_seconds": 20, "move_radius": 1}}


def strip_timing(rows):
    return [{k: v for k, v in r.items() if k != "elapsed_seconds"} for r in rows]


class RewardModeTests(unittest.TestCase):
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
        record, _ = trainer._rollout(trainer.policy, 4242, "BLUE", training=True)
        self.assertIsNotNone(record["own_score"], "合成局应当能跑完（否则本测试无意义）")
        expected = ((record["own_score"] - record["opening_own"])
                    - (record["other_score"] - record["opening_other"]))
        self.assertEqual(record["net_gain"], expected)
        self.assertEqual(record["learning_return"], expected)
        self.assertEqual(record["reward_mode"], "net_gain")

    def test_own_gain_mode_keeps_old_semantics(self):
        trainer = self.trainer(reward_mode="own_gain")
        record, _ = trainer._rollout(trainer.policy, 4242, "RED", training=True)
        self.assertEqual(record["learning_return"], record["own_gain"])
        self.assertEqual(record["own_gain"],
                         record["own_score"] - record["opening_own"])

    def test_net_gain_penalises_enemy_capture(self):
        """净胜分增量必须对“敌方涨分”敏感（旧口径完全无感）。"""
        trainer = self.trainer()
        record, _ = trainer._rollout(trainer.policy, 4243, "BLUE", training=True)
        if record["net_gain"] is None:
            self.skipTest("本局未完成或缺少开局分")
        own_move = record["own_score"] - record["opening_own"]
        enemy_move = record["other_score"] - record["opening_other"]
        self.assertEqual(record["net_gain"], own_move - enemy_move)
        # 反证：只扣本方分的旧口径看不见 enemy_move
        self.assertNotEqual(record["own_gain"], record["net_gain"]) if enemy_move else None

    def test_stride_subsamples_gradient_only(self):
        base = self.trainer()
        strided = self.trainer(trajectory_stride=4)
        base_records = base.train(1)
        strided_records = strided.train(1)
        self.assertEqual(strip_timing(base_records), strip_timing(strided_records))
        n_base = len(base.pending[0][0])
        n_strided = len(strided.pending[0][0])
        self.assertGreater(n_base, 0)
        self.assertEqual(n_strided, len(range(0, n_base, 4)))
        self.assertEqual(base.pending[0][1], strided.pending[0][1])   # 学习目标不变

    def test_resume_rejects_reward_mode_change(self):
        trainer = self.trainer()
        trainer.train(2)
        path = Path(self.tmp.name) / "ckpt.json"
        trainer.save(path)
        data = json.loads(path.read_text())
        self.assertEqual(data["reward_mode"], "net_gain")
        self.assertEqual(data["trajectory_stride"], 1)
        with self.assertRaises(SystemExit):
            main(["--snapshot", str(self.snapshot), "--resume", str(path),
                  "--reward-mode", "own_gain",
                  "--checkpoint", str(Path(self.tmp.name) / "a.json"),
                  "--report", str(Path(self.tmp.name) / "a.report.json")])
        with self.assertRaises(SystemExit):
            main(["--snapshot", str(self.snapshot), "--resume", str(path),
                  "--trajectory-stride", "4",
                  "--checkpoint", str(Path(self.tmp.name) / "b.json"),
                  "--report", str(Path(self.tmp.name) / "b.report.json")])

    def test_validation(self):
        with self.assertRaises(ValueError):
            self.trainer(reward_mode="bogus")
        for bad in (0, -1, 65, 1.5, True):
            with self.assertRaises(ValueError):
                self.trainer(trajectory_stride=bad)


if __name__ == "__main__":
    unittest.main()
