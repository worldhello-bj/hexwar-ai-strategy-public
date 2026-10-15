"""Terminal-only, observation-only learning and checkpoint reproducibility."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from env.match_eval import SIDES, write_synthetic_snapshot
from env.selfplay import MonteCarloPolicy, SelfPlayTrainer, canonical, main


class SelfPlayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.snapshot = write_synthetic_snapshot(self.root / "snapshot")

    def trainer(self, **kwargs):
        return SelfPlayTrainer(self.snapshot, episode_config={"max_seconds": 20, "max_steps": 100}, **kwargs)

    def test_terminal_score_updates_both_seats_and_no_intermediate_shaping(self):
        class ScriptedEpisode:
            def __init__(self, *args, **kwargs):
                self.n = 0

            def reset(self, **kwargs):
                return self.status()

            def status(self):
                done = self.n == 4
                return {"current_actor": None if done else SIDES[self.n % 2],
                        "terminated": done, "truncated": False,
                        "final_scores": {"BLUE": 7, "RED": -3} if done else None,
                        # Deliberately large intermediate values must never train.
                        "rewards": {"BLUE": 999, "RED": 999} if not done else {"BLUE": 7, "RED": -3}}

            def observe(self, side):
                return {"seat": side, "visible": []}

            def legal_actions(self, side):
                return [{"token": 0}]

            def step(self, action, **kwargs):
                self.n += 1
                return self.status()

        with patch("env.selfplay.TwoPlayerEpisode", ScriptedEpisode):
            trainer = self.trainer()
            trainer.train(2)
        for side, score in (("BLUE", 7), ("RED", -3)):
            entries = [entry for actions in trainer.policies[side].table.values() for entry in actions.values()]
            self.assertEqual(entries, [{"count": 2, "value": score}])

    def test_real_episode_populates_table_with_own_terminal_totals(self):
        trainer = self.trainer(seed=3, epsilon=1)
        records = trainer.train(3)
        self.assertTrue(all(r["status"]["terminated"] for r in records))
        for side in SIDES:
            entries = [e for actions in trainer.policies[side].table.values() for e in actions.values()]
            self.assertTrue(entries)
            self.assertTrue(all(e["count"] > 0 for e in entries))
            self.assertEqual({e["value"] for e in entries},
                             {r["status"]["final_scores"][side] for r in records})

    def test_truncation_never_updates(self):
        trainer = SelfPlayTrainer(self.snapshot, episode_config={"max_seconds": 999, "max_steps": 2})
        records = trainer.train(2)
        self.assertTrue(all(r["status"]["truncated"] for r in records))
        self.assertEqual({s: p.table for s, p in trainer.policies.items()}, {s: {} for s in SIDES})

    def test_checkpoint_resume_matches_uninterrupted(self):
        whole = self.trainer(seed=71)
        expected = whole.train(7)
        partial = self.trainer(seed=71)
        first = partial.train(3)
        checkpoint = self.root / "checkpoint.json"
        partial.save(checkpoint)
        resumed = SelfPlayTrainer.load(checkpoint, self.snapshot)
        self.assertEqual(first + resumed.train(4), expected)
        self.assertEqual(resumed.checkpoint(), whole.checkpoint())
        with self.assertRaises(FileExistsError):
            resumed.save(checkpoint)

    def test_evaluation_is_fresh_deterministic_and_read_only(self):
        trainer = self.trainer(seed=14)
        trainer.train(3)
        before = trainer.checkpoint()
        self.assertEqual(trainer.evaluate([90, 91]), trainer.evaluate([90, 91]))
        self.assertEqual(trainer.checkpoint(), before)
        with self.assertRaises(ValueError):
            trainer.evaluate([1])
        with self.assertRaises(ValueError):
            trainer.evaluate([4, 4])

    def test_fog_only_observation_keys_and_detached_actions(self):
        units_path = self.snapshot / "units.json"
        units = json.loads(units_path.read_text())
        units[1]["pos"] = 999999
        units_path.write_text(json.dumps(units))
        trainer = SelfPlayTrainer(self.snapshot, episode_config={"max_seconds": 1, "max_steps": 10})
        trainer.train(1)
        for actions in trainer.policies["BLUE"].table:
            state = json.loads(actions)
            self.assertEqual(set(state["score"]), {"own", "enemy", "diff"})
            self.assertTrue(all(isinstance(v, (int, float)) for v in state["score"].values()))
            self.assertEqual(state["enemy_visible"], [])
            self.assertEqual(state["enemy_known"], [])
            self.assertNotIn("999999", actions)
        policy = MonteCarloPolicy(seed=1)
        observation, legal = {"only_visible": 1}, [{"x": [1]}]
        before = copy.deepcopy((observation, legal))
        result = policy(observation, legal)
        result["x"].append(2)
        self.assertEqual((observation, legal), before)

    def test_generic_learning_changes_choice_without_action_preferences(self):
        policy = MonteCarloPolicy(seed=3, epsilon=0)
        observation = {"opaque": [1, 2]}
        legal = [{"label": "a"}, {"label": "b"}]
        policy.update([(canonical(observation), canonical(legal[1]))], 17)
        self.assertEqual(policy(observation, legal), legal[1])
        self.assertEqual(policy.table[canonical(observation)][canonical(legal[1])]["value"], 17)
        policy.update([(canonical(observation), canonical(legal[1]))], 3)
        self.assertEqual(policy.table[canonical(observation)][canonical(legal[1])]["value"], 10)

    def test_initial_terminal_has_no_fabricated_decisions(self):
        trainer = SelfPlayTrainer(self.snapshot, episode_config={"max_seconds": 0})
        result = trainer.train(1)[0]
        self.assertTrue(result["status"]["terminated"])
        self.assertEqual(result["decisions"], {s: 0 for s in SIDES})
        self.assertFalse(any(p.table for p in trainer.policies.values()))

    def test_configuration_and_checkpoint_validation(self):
        for value in (-1, 1.1, float("nan"), True):
            with self.assertRaises(ValueError):
                self.trainer(epsilon=value)
        trainer = self.trainer()
        with self.assertRaises(ValueError):
            trainer.train(10001)
        path = self.root / "checkpoint.json"
        trainer.save(path)
        units_path = self.snapshot / "units.json"
        units_path.write_text(units_path.read_text() + "\n")
        with self.assertRaises(ValueError):
            SelfPlayTrainer.load(path, self.snapshot)
        with self.assertRaises(ValueError):
            trainer.train(1)

    def test_cli_train_and_resume(self):
        first, second, report = [self.root / n for n in ("a.json", "b.json", "report.json")]
        self.assertEqual(main(["--episodes", "2", "--checkpoint", str(first), "--report", str(report)]), 0)
        self.assertEqual(main(["--episodes", "1", "--resume", str(first), "--checkpoint", str(second)]), 0)
        self.assertEqual(json.loads(second.read_text())["episodes_completed"], 3)
        self.assertEqual(len(json.loads(report.read_text())["evaluation"]), 2)


if __name__ == "__main__":
    unittest.main()
