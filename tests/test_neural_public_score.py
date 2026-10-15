"""Public-only, per-decision residual critic; terminal returns remain untouched."""
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from env.match_eval import write_synthetic_snapshot
from env.neural_policy import Decision, NeuralPolicy
from env.neural_train import NeuralTrainer


class PublicScoreBaselineTests(unittest.TestCase):
    def policy(self, **kwargs):
        return NeuralPolicy(buckets=8, hidden_dim=2, baseline_offset="public_own_score", **kwargs)

    def decisions(self, p):
        return [p._decision({"score": {"own": s}}, [1, 2, 3], i)
                for i, s in enumerate((234, 354))]

    def test_all_parameter_derivatives_and_ppo_clip_branches(self):
        p = self.policy(update_mode="ppo_clip")
        examples = list(zip(self.decisions(p), (237.5, 352.5)))
        for adv, ratio in ((2., 1.), (-2., 1.), (2., 1.4), (-2., .6), (2., .6), (-2., 1.4)):
            old = [float(p._forward(d)[5][d.selected]) - math.log(ratio) for d, _ in examples]
            for old_probs in (None, old):
                def compute():
                    return p._loss_gradients(examples, [adv] * 2, actor_normalizer=3,
                                             old_log_probs=old_probs)
                _, gradients = compute()
                for name, parameter in p.params.items():
                    for index in np.ndindex(parameter.shape):
                        value = parameter[index]
                        parameter[index] = value + 1e-5
                        plus = compute()[0]
                        parameter[index] = value - 1e-5
                        minus = compute()[0]
                        parameter[index] = value
                        self.assertAlmostEqual(gradients[name][index], (plus - minus) / 2e-5, places=7)
                if old_probs is not None and ((adv > 0 and ratio > 1.2) or (adv < 0 and ratio < .8)):
                    np.testing.assert_array_equal(gradients["policy"], 0)
                    np.testing.assert_array_equal(gradients["action"], 0)

    def test_constant_offset_invariance_without_changing_features(self):
        # Isolate baseline arithmetic: adding C to BOTH raw return and public
        # score leaves targets/advantages invariant. Re-encoding a changed public
        # observation is not invariant and is deliberately not claimed to be.
        for mode in ("reinforce", "ppo_clip"):
            p, shifted = self.policy(update_mode=mode), self.policy(update_mode=mode)
            ds = self.decisions(p)
            original = [([ds[0]], 237.), ([ds[1]], 352.)]
            translated = [([replace(d, public_score_offset=d.public_score_offset + c)], r + c)
                          for d, r, c in zip(ds, (237., 352.), (1024., -512.))]
            self.assertEqual(p.update_batch(original), shifted.update_batch(translated))
            self.assertEqual(p.checkpoint(), shifted.checkpoint())
            self.assertEqual([r for _, r in original], [237., 352.])

    def test_frozen_per_decision_offsets_and_raw_returns(self):
        p = self.policy(update_mode="ppo_clip", update_epochs=3)
        ds = self.decisions(p)
        old = [p._forward(d) for d in ds]
        captures = []
        original = p._loss_gradients
        def capture(examples, advantages, **kwargs):
            captures.append((copy.deepcopy(examples), advantages, kwargs))
            return original(examples, advantages, **kwargs)
        with patch.object(p, "_loss_gradients", side_effect=capture):
            p.update_batch([([ds[0], ds[1]], 360.)])
        for examples, advantages, kwargs in captures:
            self.assertEqual([s for _, s in examples], [360., 360.])
            self.assertEqual(advantages, (126 - old[0][4], 6 - old[1][4]))
            self.assertEqual(kwargs["baseline_offset"], 0)
        self.assertNotIn("return_baseline", p.checkpoint())

    def test_public_score_and_decision_validation(self):
        p = self.policy()
        for observation in ({}, {"score": {}}, {"score": {"own": None}},
                            {"score": {"own": True}}, {"score": {"own": float("inf")}},
                            {"score": {"own": "234"}}):
            with self.assertRaises(ValueError):
                p.sample(observation, [1, 2])
        _, d = p.sample({"score": {"own": 0}}, [1, 2])
        self.assertEqual(d.public_score_offset, 0)
        self.assertEqual(Decision.from_dict(d.to_dict()).to_dict(), d.to_dict())
        legacy = NeuralPolicy(buckets=8, hidden_dim=2)._decision({}, [1, 2])
        self.assertNotIn("public_score_offset", legacy.to_dict())
        self.assertEqual(Decision.from_dict(legacy.to_dict()).to_dict(), legacy.to_dict())
        before = p.checkpoint()
        with self.assertRaises(ValueError):
            p.update_batch([([legacy], 1)])
        self.assertEqual(before, p.checkpoint())
        for value in (None, True, "0", float("nan")):
            data = {**d.to_dict(), "public_score_offset": value}
            with self.assertRaises(ValueError):
                Decision.from_dict(data)
        with self.assertRaises(ValueError):
            p._loss_gradients([(d, 1)], baseline_offset=3)
        with self.assertRaises(ValueError):
            NeuralPolicy(buckets=8, hidden_dim=2, baseline_offset="prior_batch_mean").update_batch([([d], 1)])

    def test_pending_exact_resume_and_missing_offset_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = write_synthetic_snapshot(root / "snapshot")
            config = {"buckets": 8, "hidden_dim": 2, "baseline_offset": "public_own_score",
                      "update_mode": "ppo_clip", "update_epochs": 3}
            def trainer():
                return NeuralTrainer(snapshot, batch_episodes=2, policy_config=config)
            def rollout(t, policy, seed, side, *, training):
                _, d = policy.sample({"score": {"own": 234 if side == "BLUE" else 354}}, [1, 2])
                return {"own_score": None if seed in (2, 5) else 360}, [d]
            with patch.object(NeuralTrainer, "_rollout", rollout):
                whole, partial = trainer(), trainer()
                whole.train(8)
                partial.train(4)
                self.assertEqual(len(partial.pending), 1)
                path = root / "checkpoint.json"
                partial.save(path)
                resumed = NeuralTrainer.load(path, snapshot)
                resumed.train(4)
                self.assertEqual(whole.checkpoint(), resumed.checkpoint())
                bad = partial.checkpoint()
                del bad["pending"][0]["decisions"][0]["public_score_offset"]
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    NeuralTrainer.load(path, snapshot)
            with self.assertRaises(ValueError):
                NeuralTrainer(snapshot, policy_config=config, episode_config={"public_score": False})

    def test_cloud_runner_selection(self):
        from tools import cloud_train
        required = ["--snapshot", "/unused/snapshot", "--output", "/unused/output"]
        default = cloud_train.requested_config(cloud_train.parser().parse_args(required))
        self.assertEqual(default["policy_config"]["baseline_offset"], "prior_batch_mean")
        selected = cloud_train.requested_config(cloud_train.parser().parse_args(
            required + ["--baseline-offset", "public_own_score", "--projection-backend", "scipy_csr"]))
        self.assertEqual(selected["policy_config"]["baseline_offset"], "public_own_score")
        self.assertEqual(selected["policy_config"]["projection_backend"], "scipy_csr")

    def test_actual_public_observations_all_schemas_and_raw_report(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = write_synthetic_snapshot(Path(directory) / "snapshot")
            for schema in (1, 2, 3):
                trainer = NeuralTrainer(snapshot, observation_schema=schema,
                    policy_config={"buckets": 8, "hidden_dim": 2, "baseline_offset": "public_own_score"},
                    episode_config={"max_seconds": 20, "max_steps": 100})
                report, decisions = trainer._rollout(trainer.policy, 0, "BLUE", training=True)
                self.assertIsNotNone(report["own_score"])
                self.assertTrue(decisions)
                self.assertTrue(all(d.public_score_offset is not None for d in decisions))
                before = copy.deepcopy(report)
                trainer.policy.update_batch([(decisions, report["own_score"])])
                self.assertEqual(report, before)


if __name__ == "__main__":
    unittest.main()
