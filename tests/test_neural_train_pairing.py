# -*- coding: utf-8 -*-
"""成对训练（红一局 + 蓝一局算总分）的契约。**恒开，没有开关。**

真实比赛按**两局总分**判胜负：
- 第 i 局用种子 `train_seed_start + i//2`、席位交替（同种子、两边各打一局）；
- 回报用**两局净分之和**：R = net(蓝) + net(红) = (S蓝+S红) − (O蓝+O红)
  —— 开局席位常数（234/354）成对后恰好完全抵消，等于平台判胜负的总分差；
- 两局的动作都拿同一个配对回报（胜负由总分决定，归因应在配对级）。

半对（只打了蓝局、红局还没打）**必须留在缓冲里等下一次调用**，不能在一次
`train()` 结尾就按单局目标冲掉 —— 否则“拆成两次 train()”与“一次跑完”的
优化序列就不一致。缓冲要随检查点落盘，续跑才能接着拼对。
"""
import tempfile
import unittest
from pathlib import Path

from env.match_eval import write_synthetic_snapshot
from env.neural_train import NeuralTrainer

CONFIG = {"policy_config": {"buckets": 32, "hidden_dim": 4},
          "episode_config": {"max_seconds": 20, "move_radius": 1}}


class SeatPairingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.snapshot = write_synthetic_snapshot(Path(self.tmp.name) / "snapshot")

    def trainer(self, **kwargs):
        kwargs.setdefault("interface", "realtime")
        kwargs.setdefault("batch_episodes", 2)
        return NeuralTrainer(self.snapshot, **CONFIG, **kwargs)

    def test_slot_mapping(self):
        """同一种子打两局：先蓝后红。"""
        trainer = self.trainer()
        self.assertEqual([trainer._slot(i) for i in range(4)],
                         [(0, "BLUE"), (0, "RED"), (1, "BLUE"), (1, "RED")])
        shifted = self.trainer(train_seed_start=10)
        self.assertEqual([shifted._slot(i) for i in range(2)],
                         [(10, "BLUE"), (10, "RED")])

    def test_pair_return_is_sum_of_both_seats(self):
        trainer = self.trainer()
        seen = {}
        trainer.policy.update_batch = lambda batch: seen.setdefault(
            "targets", [target for _, target in batch]) and \
            {"decisions": 0, "loss": 0.0, "gradient_norm": 0.0}
        records = trainer.train(2)
        expected = records[0]["net_gain"] + records[1]["net_gain"]
        self.assertEqual({r["seed"] for r in records}, {0})            # 同一种子
        self.assertEqual([r["learner_side"] for r in records], ["BLUE", "RED"])
        self.assertAlmostEqual(expected, records[0]["pair_return"])
        self.assertAlmostEqual(expected, records[1]["pair_return"])
        # 两局进梯度时用的是同一个配对回报
        self.assertEqual(len(seen["targets"]), 2)
        self.assertAlmostEqual(seen["targets"][0], expected)
        self.assertAlmostEqual(seen["targets"][1], expected)
        self.assertEqual(trainer.updates, 1)

    def test_half_pair_waits_in_buffer(self):
        """只打了蓝局时：不进 pending，留在缓冲里等红局。"""
        trainer = self.trainer()
        trainer.train(1)
        self.assertEqual(list(trainer._pair_buffer), [0])
        self.assertEqual(list(trainer._pair_buffer[0]), ["BLUE"])
        self.assertEqual(trainer.pending, [])
        self.assertEqual(trainer.episodes_completed, 1)
        self.assertEqual(trainer.updates, 0)

    def test_half_pair_survives_checkpoint_and_completes_on_resume(self):
        trainer = self.trainer()
        trainer.train(1)
        path = Path(self.tmp.name) / "half.json"
        trainer.save(path)
        reloaded = NeuralTrainer.load(path, self.snapshot)
        self.assertEqual(list(reloaded._pair_buffer), [0])
        self.assertEqual([r["learner_side"] for r in reloaded.train(1)], ["RED"])
        self.assertEqual(reloaded._pair_buffer, {})
        self.assertEqual(reloaded.updates, 1)
        self.assertEqual(reloaded.episodes_completed, 2)

    def test_odd_batch_rejected(self):
        with self.assertRaises(ValueError):
            self.trainer(batch_episodes=3)


if __name__ == "__main__":
    unittest.main()
