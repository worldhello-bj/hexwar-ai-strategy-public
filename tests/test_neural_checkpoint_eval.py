"""Fresh evaluation is bounded, paired, disjoint, and cannot train models."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from env.match_eval import SIDES, write_synthetic_snapshot
from env.neural_train import NeuralTrainer
from tools.evaluate_neural_checkpoint import evaluate_checkpoint, main, paired_deltas


class NeuralCheckpointEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.snapshot = write_synthetic_snapshot(self.root / "snapshot")
        self.trainer = NeuralTrainer(self.snapshot, train_seed_start=10, batch_episodes=2,
            policy_config={"buckets": 16, "hidden_dim": 4},
            episode_config={"max_seconds": 10, "max_steps": 20}, heldout_seeds=[90])
        self.trainer.train(2)
        self.path = self.root / "checkpoint.json"
        self.trainer.save(self.path)

    def test_fresh_seeds_source_bytes_hash_and_loaded_models_unchanged(self):
        source = self.path.read_bytes()
        frozen = copy.deepcopy(self.trainer.checkpoint())
        # Capture the independent loaded object to assert full immutability,
        # apart from the explicitly substituted in-memory seed reservation.
        clone = NeuralTrainer.load(self.path, self.snapshot)
        expected = clone.checkpoint()
        expected["heldout_seeds"] = [100, 101]
        with patch("tools.evaluate_neural_checkpoint.NeuralTrainer.load", return_value=clone), \
                patch.object(clone.policy, "update_batch", side_effect=AssertionError("evaluation trained")):
            report = evaluate_checkpoint(self.path, self.snapshot, seed_start=100, seed_count=2)
        self.assertEqual(clone.checkpoint(), expected)
        self.assertEqual(self.trainer.checkpoint(), frozen)
        self.assertEqual(self.path.read_bytes(), source)
        self.assertEqual(report["source_checkpoint_sha256"], hashlib.sha256(source).hexdigest())
        self.assertEqual(report["fresh_heldout_seeds"], [100, 101])
        self.assertEqual(report["paired_comparison"]["complete_seed_pairs"], 2)
        for label in ("initial", "trained"):
            self.assertEqual([(r["seed"], r["learner_side"]) for r in report["evaluation"][label]["games"]],
                             [(100, "BLUE"), (100, "RED"), (101, "BLUE"), (101, "RED")])

    def test_overlap_rejected_before_mutating_reservation_or_running(self):
        for start in (9, 10, 11, 89, 90):
            clone = NeuralTrainer.load(self.path, self.snapshot)
            before = clone.checkpoint()
            with patch("tools.evaluate_neural_checkpoint.NeuralTrainer.load", return_value=clone), \
                    patch.object(clone, "evaluate", side_effect=AssertionError("must reject first")):
                with self.assertRaises(ValueError):
                    evaluate_checkpoint(self.path, self.snapshot, seed_start=start, seed_count=2)
            self.assertEqual(clone.checkpoint(), before)

    def test_invalid_budgets(self):
        for count in (0, -1, 257, True):
            with self.assertRaises(ValueError):
                evaluate_checkpoint(self.path, self.snapshot, seed_start=100, seed_count=count)

    def test_seed_level_pair_statistic_and_truncation_exclusion(self):
        evaluation = {label: {"games": []} for label in ("initial", "trained")}
        for seed, deltas in ((1, [2, 4]), (2, [-2, 0]), (3, [None, 4])):
            for side, delta in zip(SIDES, deltas):
                evaluation["initial"]["games"].append({"seed": seed, "learner_side": side, "own_score": 10})
                evaluation["trained"]["games"].append({"seed": seed, "learner_side": side,
                                                       "own_score": 10 + delta if delta is not None else None})
        result = paired_deltas(evaluation, [1, 2, 3])
        self.assertEqual(result["mean_own_score_delta"], 1)
        self.assertEqual(result["sample_standard_error"], 2)
        self.assertEqual(result["excluded_seed_pairs"], 1)
        self.assertEqual([r["mean_own_score_delta"] for r in result["pairs"]], [3, -1, None])

    def test_cli_writes_new_report_only(self):
        output = self.root / "report.json"
        source = self.path.read_bytes()
        self.assertEqual(main(["--checkpoint", str(self.path), "--snapshot", str(self.snapshot),
                               "--seed-start", "100", "--seed-count", "1", "--report", str(output)]), 0)
        self.assertEqual(self.path.read_bytes(), source)
        self.assertTrue(json.loads(output.read_text())["models_and_source_checkpoint_unchanged"])


class PublicNeuralCheckpointEvaluationTests(NeuralCheckpointEvaluationTests):
    def setUp(self):
        super().setUp()
        self.trainer = NeuralTrainer(self.snapshot, observation_schema=2,
            train_seed_start=10, batch_episodes=2, policy_config={"buckets": 16, "hidden_dim": 4},
            episode_config={"max_seconds": 10, "max_steps": 20}, heldout_seeds=[90])
        self.trainer.train(2)
        self.path = self.root / "public-checkpoint.json"
        self.trainer.save(self.path)

    def test_frozen_evaluator_reports_checkpoint_schema(self):
        report = evaluate_checkpoint(self.path, self.snapshot, seed_start=100, seed_count=1)
        self.assertEqual(report["observation_schema"], 2)
        self.assertEqual(report["policy_config"]["representation"], "public_v2_split")
        self.assertTrue(report["models_and_source_checkpoint_unchanged"])


if __name__ == "__main__":
    unittest.main()
