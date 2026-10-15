"""Tiny synthetic held-out evaluation, seat routing and read-only contracts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from env.match_eval import SIDES, write_synthetic_snapshot
from env.policy_eval import MAX_SEEDS, evaluate_checkpoint, main
from env.selfplay import MonteCarloPolicy, SelfPlayTrainer


class PolicyEvalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.snapshot = write_synthetic_snapshot(self.root / "snapshot")
        self.checkpoint = self.root / "checkpoint.json"
        self.trainer = SelfPlayTrainer(self.snapshot, episode_config={"max_seconds": 20, "max_steps": 20})
        self.trainer.train(2)
        self.trainer.save(self.checkpoint)

    def evaluate(self, seeds=(100, 101)):
        return evaluate_checkpoint(self.checkpoint, self.snapshot, seeds)

    def test_both_seats_deterministic_read_only_and_uniform_baseline(self):
        original = self.checkpoint.read_bytes()
        with patch.object(MonteCarloPolicy, "update", side_effect=AssertionError("no training")):
            first = self.evaluate()
            self.assertEqual(first, self.evaluate())
        self.assertEqual(original, self.checkpoint.read_bytes())
        self.assertEqual([(r["seed"], r["policy_side"]) for r in first["games"]],
                         [(100, "BLUE"), (100, "RED"), (101, "BLUE"), (101, "RED")])
        self.assertEqual(first["summary"]["completed"], 4)
        self.assertEqual(first["summary"]["draw"], 4)
        self.assertEqual(first["summary"]["rates"]["draw"], 1)

    def test_seed_validation_and_manifest(self):
        for seeds in ([], [0], [100, 100], [True], [-1], list(range(100, 100 + MAX_SEEDS + 1))):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                self.evaluate(seeds)
        path = self.snapshot / "cities.json"
        path.write_text("[ ]")
        with self.assertRaisesRegex(ValueError, "snapshot"):
            self.evaluate()

    def test_truncation_has_no_outcome_or_rates(self):
        data = json.loads(self.checkpoint.read_text())
        data["episode_config"]["max_steps"] = 0
        self.checkpoint.write_text(json.dumps(data))
        report = self.evaluate()
        self.assertEqual(report["summary"]["truncated"], 4)
        self.assertEqual(report["summary"]["completed"], 0)
        self.assertTrue(all(value is None for value in report["summary"]["rates"].values()))
        self.assertTrue(all(r["result"] is None and r["status"]["final_scores"] is None
                            for r in report["games"]))

    def test_correct_table_and_fresh_policy_per_seat_assignment(self):
        actual = MonteCarloPolicy
        constructed = []

        def capture(**kwargs):
            policy = actual(**kwargs)
            constructed.append((kwargs, policy))
            return policy

        with patch("env.policy_eval.MonteCarloPolicy", side_effect=capture):
            self.evaluate()
        self.assertEqual(len(constructed), 4)
        for index, (kwargs, policy) in enumerate(constructed):
            seat = SIDES[index % 2]
            self.assertEqual(kwargs["table"], self.trainer.policies[seat].table)
            self.assertEqual(kwargs["epsilon"], 0)
        self.assertEqual(len({kwargs["seed"] for kwargs, _ in constructed}), 4)
        self.assertEqual(len({id(policy) for _, policy in constructed}), 4)

    def test_terminal_score_wins_not_status_winner(self):
        class TerminalEpisode:
            def __init__(self, *args, **kwargs):
                pass

            def reset(self, **kwargs):
                return {"terminated": True, "truncated": False,
                        "winner": "RED", "final_scores": {"BLUE": 9, "RED": 2}}

        with patch("env.policy_eval.TwoPlayerEpisode", TerminalEpisode):
            report = self.evaluate([100])
        self.assertEqual([r["result"] for r in report["games"]], ["win", "loss"])
        self.assertEqual(report["summary"]["rates"], {"win": .5, "draw": 0, "loss": .5})

    def test_checkpoint_validation_and_public_score(self):
        original = json.loads(self.checkpoint.read_text())
        for change in ("schema", "table", "public", "steps"):
            data = json.loads(json.dumps(original))
            if change == "schema":
                data["schema_version"] = 999
            elif change == "table":
                del data["tables"]["RED"]
            elif change == "public":
                data["episode_config"]["public_score"] = False
            else:
                data["episode_config"]["max_steps"] = 10001
            self.checkpoint.write_text(json.dumps(data))
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.evaluate()

    def test_cli_new_report_and_no_overwrite(self):
        report = self.root / "report.json"
        argv = ["--checkpoint", str(self.checkpoint), "--snapshot", str(self.snapshot),
                "--seeds", "100", "--report", str(report)]
        self.assertEqual(main(argv), 0)
        self.assertEqual(json.loads(report.read_text())["summary"]["games"], 2)
        before = report.read_bytes()
        with self.assertRaises(SystemExit):
            main(argv)
        self.assertEqual(report.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
