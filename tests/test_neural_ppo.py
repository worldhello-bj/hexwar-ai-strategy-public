"""Clipped surrogate derivatives, frozen batches, strict resume and legacy bytes."""
import copy
import io
import json
import math
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from env.match_eval import write_synthetic_snapshot
from env.neural_policy import NeuralPolicy
from env.neural_train import NeuralTrainer, canonical, main
from tests.fixtures.pre_ppo_neural_policy import NeuralPolicy as PrePPOPolicy


class ClippedPolicyTests(unittest.TestCase):
    def policy(self, **kwargs):
        return NeuralPolicy(buckets=8, hidden_dim=2, update_mode="ppo_clip", **kwargs)

    def test_every_parameter_derivative_both_signs_and_clip_branches(self):
        p = self.policy()
        examples = [(p._decision({"x": 2}, [1, -2, 3], 1), 3.5),
                    (p._decision([False], ["a", "b"]), -1.5)]
        for advantage, ratio in ((2., 1.), (-2., 1.), (2., 1.4), (-2., .6), (2., .6), (-2., 1.4)):
            with self.subTest(advantage=advantage, ratio=ratio):
                old = [float(p._forward(d)[5][d.selected]) - math.log(ratio) for d, _ in examples]
                def compute():
                    return p._loss_gradients(examples, [advantage] * 2,
                        actor_normalizer=3, baseline_offset=.5, old_log_probs=old)
                _, gradients = compute()
                for key, parameter in p.params.items():
                    for index in np.ndindex(parameter.shape):
                        value = parameter[index]
                        parameter[index] = value + 1e-5
                        plus = compute()[0]
                        parameter[index] = value - 1e-5
                        minus = compute()[0]
                        parameter[index] = value
                        self.assertAlmostEqual(gradients[key][index], (plus - minus) / 2e-5,
                                               places=7, msg=str((key, index)))
                if (advantage > 0 and ratio > 1.2) or (advantage < 0 and ratio < .8):
                    np.testing.assert_array_equal(gradients["policy"], 0)
                    np.testing.assert_array_equal(gradients["action"], 0)

    def test_log_ratio_stability_and_exact_normalization(self):
        p = self.policy()
        d = p._decision({}, [1, 2])
        f = p._forward(d)
        for advantage, delta, expected in ((2., 10000., 2.4), (-2., -10000., -1.6),
                                            (2., -10000., 0.), (0., 10000., 0.)):
            loss, gradients = p._loss_gradients([(d, f[4])], [advantage], actor_normalizer=3,
                                              old_log_probs=[f[5][d.selected] - delta])
            self.assertAlmostEqual(loss, -expected / 3)
            self.assertTrue(all(np.isfinite(g).all() for g in gradients.values()))
        # Adverse unbounded ratios must fail, never be silently clipped away.
        with self.assertRaises(FloatingPointError):
            p._loss_gradients([(d, f[4])], [-2.], old_log_probs=[-10000.])
        # Both probabilities can underflow to zero; their log difference is valid.
        fake = list(f)
        fake[5] = np.array([-1000., 0.])
        fake[3] = np.array([0., 1.])
        with patch.object(p, "_forward", return_value=tuple(fake)):
            loss, _ = p._loss_gradients([(d, f[4])], [2.], old_log_probs=[-1000.])
        self.assertEqual(loss, -2.)

    def test_old_policy_advantages_and_offset_frozen_all_epochs(self):
        p = self.policy(update_epochs=3, baseline_offset="prior_batch_mean", learning_rate=.1)
        p.update_batch([([], 10)])
        d = p._decision({}, [1, 2])
        old = p._forward(d)
        captures = []
        original = p._loss_gradients
        def capture(examples, advantages, **kwargs):
            captures.append((advantages, kwargs.copy(), p.return_mean, p.return_count,
                             float(p._forward(d)[5][d.selected])))
            return original(examples, advantages, **kwargs)
        with patch.object(p, "_loss_gradients", side_effect=capture):
            result = p.update_batch([([d], 30), ([d, d], -2), ([], 5)])
        self.assertEqual(p.steps, 3)
        self.assertEqual(result["optimizer_steps"], 3)
        for advantages, kwargs, mean, count, _ in captures:
            self.assertEqual(advantages, (20 - old[4], -12 - old[4], -12 - old[4]))
            self.assertEqual(kwargs["old_log_probs"], (float(old[5][d.selected]),) * 3)
            self.assertEqual(kwargs["baseline_offset"], 10)
            self.assertEqual(kwargs["actor_normalizer"], 3)
            self.assertEqual((mean, count), (10, 1))
        self.assertNotEqual(captures[0][-1], captures[-1][-1])
        self.assertEqual((p.return_mean, p.return_count), (10.75, 4))

    def test_atomic_epoch_failure_and_empty_batches(self):
        p = self.policy(baseline_offset="prior_batch_mean")
        before = p.checkpoint()
        p.update_batch([])
        self.assertEqual(p.checkpoint(), before)
        d = p._decision({}, [1, 2])
        original = p._loss_gradients
        calls = []
        def fail_second(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise FloatingPointError("test failure after first epoch")
            return original(*args, **kwargs)
        with patch.object(p, "_loss_gradients", side_effect=fail_second), self.assertRaises(FloatingPointError):
            p.update_batch([([d], 42)])
        self.assertEqual(p.checkpoint(), before)
        p.update_batch([([], 7)])
        self.assertEqual((p.steps, p.return_mean, p.return_count), (0, 7, 1))

    def test_default_checkpoint_bytes_and_arithmetic_unchanged(self):
        # Compare independent pre-PPO source and current source in the SAME
        # runtime. Fixed post-update digests depend on platform/libm/BLAS last
        # bits and cannot establish arithmetic compatibility across machines.
        # Exact comparisons remain: no tolerances or rounded floating values.
        for baseline_offset in ("none", "prior_batch_mean"):
            with self.subTest(baseline_offset=baseline_offset):
                config = dict(seed=17, buckets=8, hidden_dim=2, baseline_offset=baseline_offset)
                current, reference = NeuralPolicy(**config), PrePPOPolicy(**config)
                self.assertEqual(canonical(current.checkpoint()), canonical(reference.checkpoint()))
                for score in (12, -3, 0):
                    action, decision = current.sample({"a": 2}, [1, 2, 3])
                    old_action, old_decision = reference.sample({"a": 2}, [1, 2, 3])
                    self.assertEqual(action, old_action)
                    self.assertEqual(decision.to_dict(), old_decision.to_dict())
                    result = current.update_batch([([decision, decision], score), ([], 8)])
                    old_result = reference.update_batch([([old_decision, old_decision], score), ([], 8)])
                    self.assertEqual(result, old_result)
                    # Includes exact schema/config, all tensors, optimizer state,
                    # step counters, RNG state, and optional baseline state.
                    self.assertEqual(canonical(current.checkpoint()), canonical(reference.checkpoint()))
                with tempfile.TemporaryDirectory() as directory:
                    current_path, old_path = Path(directory) / "current.json", Path(directory) / "old.json"
                    current.save(current_path)
                    reference.save(old_path)
                    self.assertEqual(current_path.read_bytes(), old_path.read_bytes())
                self.assertEqual(NeuralPolicy.from_checkpoint(reference.checkpoint()).checkpoint(),
                                 current.checkpoint())

    def test_config_and_policy_checkpoint_strictness(self):
        for kwargs in ({"update_mode": "bad"}, {"ppo_clip": .2}, {"update_epochs": 2},
                       {"update_mode": "ppo_clip", "ppo_clip": 0},
                       {"update_mode": "ppo_clip", "ppo_clip": float("nan")},
                       {"update_mode": "ppo_clip", "ppo_clip": 1},
                       {"update_mode": "ppo_clip", "update_epochs": True},
                       {"update_mode": "ppo_clip", "update_epochs": 1.5}):
            with self.assertRaises(ValueError):
                NeuralPolicy(**kwargs)
        for mode in ("reinforce", "ppo_clip"):
            data = NeuralPolicy(buckets=8, hidden_dim=2, update_mode=mode).checkpoint()
            data["algorithm"] = "generic_json_ppo_clip" if mode == "reinforce" else "generic_json_reinforce"
            with self.assertRaises(ValueError):
                NeuralPolicy.from_checkpoint(data)

    def test_ppo_checkpoint_requires_all_settings_and_complete_epochs(self):
        data = self.policy(update_epochs=3).checkpoint()
        for key in ("update_mode", "ppo_clip", "update_epochs"):
            malformed = copy.deepcopy(data)
            del malformed["config"][key]
            with self.assertRaises(ValueError):
                NeuralPolicy.from_checkpoint(malformed)
        data["steps"] = 1
        with self.assertRaises(ValueError):
            NeuralPolicy.from_checkpoint(data)

    def test_trainer_resume_pending_truncations_and_frozen_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = write_synthetic_snapshot(root / "snapshot")
            def trainer():
                return NeuralTrainer(snapshot, batch_episodes=2, policy_config={"buckets": 8,
                    "hidden_dim": 2, "update_mode": "ppo_clip", "update_epochs": 3,
                    "baseline_offset": "prior_batch_mean"})
            seen = []
            def rollout(t, policy, seed, side, *, training):
                seen.append((seed, copy.deepcopy(policy.checkpoint()["params"]), policy.steps))
                score = None if seed in (2, 5) else seed * 10
                _, decision = policy.sample({"s": seed}, [1, 2])
                return {"own_score": score}, [decision]
            with patch.object(NeuralTrainer, "_rollout", rollout):
                whole = trainer()
                whole.train(8)
                # 成对：同一对（种子）的两局之间不会有更新，配对完成后参数才变
                self.assertEqual(seen[0][1], seen[1][1])
                self.assertEqual(seen[2][1], seen[3][1])
                self.assertEqual(seen[4][1], seen[5][1])
                self.assertNotEqual(seen[3][1], seen[4][1])
                partial = trainer()
                partial.train(4)
                self.assertEqual(len(partial.pending), 0)
                path = root / "saved.json"
                partial.save(path)
                resumed = NeuralTrainer.load(path, snapshot)
                resumed.train(4)
                self.assertEqual(resumed.checkpoint(), whole.checkpoint())
                # 3 次配对更新，每次 3 个 epoch
                self.assertEqual((resumed.updates, resumed.policy.steps, resumed.policy.return_count), (3, 9, 6))
            # A failed final rollout/update restores a resumable partial batch,
            # including its sampling RNG, rather than leaving a full pending list.
            with patch.object(NeuralTrainer, "_rollout", rollout):
                retry = trainer()
                retry.train(1)
                before = retry.checkpoint()
                original = retry.policy._loss_gradients
                calls = []
                def fail_second(*args, **kwargs):
                    calls.append(1)
                    if len(calls) == 2:
                        raise FloatingPointError("failed second epoch")
                    return original(*args, **kwargs)
                with patch.object(retry.policy, "_loss_gradients", side_effect=fail_second), self.assertRaises(FloatingPointError):
                    retry.train(1)
                self.assertEqual(retry.checkpoint(), before)
                retry_path = root / "retry.json"
                retry.save(retry_path)
                retry = NeuralTrainer.load(retry_path, snapshot)
                retry.train(7)
                self.assertEqual(retry.checkpoint(), whole.checkpoint())
            # partial 已做 2 次更新（每次 3 epoch）⇒ 合法 steps 只有 0/3/6
            for steps in (1, 4, 7):
                data = partial.checkpoint()
                data["policy"]["steps"] = steps
                bad = root / "bad.json"
                bad.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    NeuralTrainer.load(bad, snapshot)
            # No CLI override may silently convert a resumed learner.
            for extra in (["--update-mode", "reinforce"], ["--update-epochs", "4"], ["--ppo-clip", ".1"]):
                with self.assertRaises(SystemExit), patch.object(NeuralTrainer, "train") as train, redirect_stderr(io.StringIO()):
                    main(["--snapshot", str(snapshot), "--resume", str(path), "--checkpoint", str(root / "out.json"),
                          "--report", str(root / "report.json"), *extra])
                train.assert_not_called()
            with patch.object(NeuralTrainer, "train", return_value=[]), patch.object(NeuralTrainer, "evaluate",
                    return_value={"initial": {"summary": {}}, "trained": {"summary": {}}}), redirect_stdout(io.StringIO()):
                main(["--snapshot", str(snapshot), "--resume", str(path), "--checkpoint", str(root / "out.json"),
                      "--report", str(root / "report.json")])
            self.assertEqual(json.loads((root / "out.json").read_text()), partial.checkpoint())


if __name__ == "__main__":
    unittest.main()
