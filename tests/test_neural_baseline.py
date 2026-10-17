"""Historical return offset: detached gradients, exact state and opt-in only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from env.match_eval import write_synthetic_snapshot
from env.neural_policy import NeuralPolicy
from env.neural_train import NeuralTrainer, main


class HistoricalBaselineTests(unittest.TestCase):
    def policy(self):
        return NeuralPolicy(buckets=8, hidden_dim=2, baseline_offset="prior_batch_mean")

    def test_constant_offset_every_parameter_finite_difference(self):
        p = self.policy()
        examples = [(p._decision({"a": 2}, [1, -2], 1), 33.5),
                    (p._decision([None, False], ["a", "b", "c"]), 28.8)]
        offset = 30.0
        advantages = [score - offset - p._forward(d)[4] for d, score in examples]
        def compute():
            return p._loss_gradients(examples, advantages, actor_normalizer=3, baseline_offset=offset)
        _, gradients = compute()
        for name, parameter in p.params.items():
            for index in np.ndindex(parameter.shape):
                old = parameter[index]
                parameter[index] = old + 1e-5
                plus = compute()[0]
                parameter[index] = old - 1e-5
                minus = compute()[0]
                parameter[index] = old
                self.assertAlmostEqual(gradients[name][index], (plus - minus) / 2e-5,
                                       places=7, msg=str((name, index)))

    def test_current_returns_never_enter_preupdate_offset(self):
        p = self.policy()
        d = p._decision({}, [1, 2])
        original = p._loss_gradients
        seen = []
        def capture(examples, **kwargs):
            seen.append((kwargs["baseline_offset"], p.return_mean, p.return_count,
                         [score for _, score in examples]))
            return original(examples, **kwargs)
        with patch.object(p, "_loss_gradients", side_effect=capture):
            p.update_batch([([d], 20), ([d, d], 40)])
            p.update_batch([([d], 1000)])
        self.assertEqual(seen, [(0.0, 0.0, 0, [20, 40, 40]), (30.0, 30.0, 2, [1000])])
        self.assertEqual(p.return_count, 3)
        self.assertAlmostEqual(p.return_mean, 1060 / 3)

    def test_empty_zero_no_decisions_and_failure(self):
        p = self.policy()
        before = p.checkpoint()
        self.assertEqual(p.update_batch([])["decisions"], 0)
        self.assertEqual(p.checkpoint(), before)
        p.update_batch([([], 0), ([], 60)])
        self.assertEqual((p.return_mean, p.return_count, p.steps), (30, 2, 0))
        self.assertEqual(p.checkpoint()["params"], before["params"])
        before = p.checkpoint()
        with self.assertRaises(ValueError):
            p.update_batch([([], float("nan"))])
        self.assertEqual(p.checkpoint(), before)
        d = p._decision({}, [1, 2])
        with patch.object(p, "_loss_gradients", return_value=(float("nan"), {k: np.zeros_like(v) for k, v in p.params.items()})):
            with self.assertRaises(FloatingPointError):
                p.update_batch([([d], 100)])
        self.assertEqual(p.checkpoint(), before)

    def test_statistics_partition_and_json_resume_exact(self):
        scores = [0, 1.25, 700, -2, 38.4, 1e8, -3, 19]
        whole, partial = self.policy(), self.policy()
        whole.update_batch([([], s) for s in scores])
        partial.update_batch([([], s) for s in scores[:3]])
        partial = NeuralPolicy.from_checkpoint(json.loads(json.dumps(partial.checkpoint())))
        for score in scores[3:]:
            partial.update_batch([([], score)])
        self.assertEqual(partial.checkpoint(), whole.checkpoint())
        # Actual optimizer and RNG resume must also agree across identical batches.
        d = whole._decision({}, [1, 2])
        whole.update_batch([([d], 27)])
        partial.update_batch([([d], 27)])
        self.assertEqual(whole.checkpoint(), partial.checkpoint())

    def test_checkpoint_validation_and_legacy_default(self):
        legacy = NeuralPolicy(buckets=8, hidden_dim=2).checkpoint()
        self.assertNotIn("baseline_offset", legacy["config"])
        self.assertNotIn("return_baseline", legacy)
        self.assertEqual(NeuralPolicy.from_checkpoint(legacy).checkpoint(), legacy)
        malformed = copy.deepcopy(legacy)
        malformed["config"]["baseline_offset"] = "prior_batch_mean"
        with self.assertRaises(ValueError):
            NeuralPolicy.from_checkpoint(malformed)
        malformed = copy.deepcopy(legacy)
        malformed["return_baseline"] = {"mean": 0, "count": 0}
        with self.assertRaises(ValueError):
            NeuralPolicy.from_checkpoint(malformed)
        valid = self.policy().checkpoint()
        for state in ({"mean": 2, "count": 0}, {"mean": 0, "count": True},
                      {"mean": float("inf"), "count": 1}, {"mean": 0, "count": -1}):
            malformed = copy.deepcopy(valid)
            malformed["return_baseline"] = state
            with self.assertRaises(ValueError):
                NeuralPolicy.from_checkpoint(malformed)

    def test_trainer_partial_batch_resume_truncation_and_cli_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = write_synthetic_snapshot(root / "snapshot")
            def trainer():
                return NeuralTrainer(snapshot, batch_episodes=2, policy_config={
                    "buckets": 8, "hidden_dim": 2, "baseline_offset": "prior_batch_mean"})
            def rollout(t, policy, seed, side, *, training):
                # 成对训练：种了下标 i//2 ⇒ 0..3 每个种子打蓝/红两局。
                # 包含已完成但零决策的局（seed 0）、零分局与被截断的局（seed 2）。
                score = None if seed == 2 else seed * 10
                decisions = [] if seed == 0 else [policy._decision({"s": seed}, [1, 2])]
                return {"own_score": score}, decisions
            with patch.object(NeuralTrainer, "_rollout", rollout):
                whole = trainer()
                whole.train(8)
                partial = trainer()
                partial.train(4)
                # 种子 0/1 两对已经更新过（每对 2 局）
                self.assertEqual(partial.policy.return_count, 4)
                self.assertEqual(len(partial.pending), 0)
                path = root / "saved.json"
                partial.save(path)
                resumed = NeuralTrainer.load(path, snapshot)
                resumed.train(4)
                self.assertEqual(resumed.checkpoint(), whole.checkpoint())
                # 种子 2 被截断（不入批），种子 3 贡献两局
                self.assertEqual(resumed.policy.return_count, 6)
                self.assertAlmostEqual(resumed.policy.return_mean, 80.0 / 6.0)
                before = resumed.checkpoint()
                resumed.train(0)
                self.assertEqual(before, resumed.checkpoint())
            # Omitting the flag inherits the saved mode without conversion.
            with patch.object(NeuralTrainer, "train", return_value=[]), patch.object(
                    NeuralTrainer, "evaluate", return_value={"initial": {"summary": {}}, "trained": {"summary": {}}}):
                main(["--snapshot", str(snapshot), "--resume", str(path),
                      "--checkpoint", str(root / "inherited.json"), "--report", str(root / "inherited-report.json")])
            self.assertEqual(json.loads((root / "inherited.json").read_text()), partial.checkpoint())
            bad = partial.checkpoint()
            bad["policy"]["return_baseline"]["count"] += 1
            bad_path = root / "bad.json"
            bad_path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                NeuralTrainer.load(bad_path, snapshot)
            for checkpoint, mode in ((path, "none"),):
                with self.assertRaises(SystemExit), patch.object(NeuralTrainer, "train") as train:
                    main(["--snapshot", str(snapshot), "--resume", str(checkpoint),
                          "--baseline-offset", mode, "--checkpoint", str(root / "out.json"),
                          "--report", str(root / "report.json")])
                train.assert_not_called()
            legacy = NeuralTrainer(snapshot, policy_config={"buckets": 8, "hidden_dim": 2})
            legacy_path = root / "legacy.json"
            legacy.save(legacy_path)
            with self.assertRaises(SystemExit), patch.object(NeuralTrainer, "train") as train:
                main(["--snapshot", str(snapshot), "--resume", str(legacy_path),
                      "--baseline-offset", "prior_batch_mean", "--checkpoint", str(root / "out.json"),
                      "--report", str(root / "report.json")])
            train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
