"""Own terminal returns, exact resume and frozen paired evaluation contracts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from env.episode_api import TwoPlayerEpisode
from env.match_eval import SIDES, write_synthetic_snapshot
from env.neural_train import NeuralTrainer, canonical, main


class NeuralTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.snapshot = write_synthetic_snapshot(self.root / "snapshot")

    def trainer(self, **kwargs):
        return NeuralTrainer(self.snapshot, policy_config={"buckets": 32, "hidden_dim": 4},
                             episode_config={"max_seconds": 20, "max_steps": 100}, **kwargs)

    def checkpoint_file(self, **kwargs):
        """写一个训练器检查点，返回 (path, trainer)。"""
        trainer = self.trainer(**kwargs)
        trainer.train(2)
        path = self.root / ("ckpt-%d.json" % len(list(self.root.glob("ckpt-*.json"))))
        trainer.save(path)
        return path, trainer

    def test_checkpoint_binds_scoring_version_and_mode(self):
        """计分口径变了就不能续跑：否则同一快照下两套奖励会混在一起。"""
        from engine import scoring
        path, trainer = self.checkpoint_file()
        data = json.loads(path.read_text())
        self.assertEqual(data["scoring_version"], scoring.SCORING_VERSION)
        self.assertEqual(data["scoring_mode"], "verified")
        self.assertEqual(data["episode_config"]["scoring_mode"], "verified")
        # ① 口径版本不同 → 拒绝
        bumped = copy.deepcopy(data)
        bumped["scoring_version"] = scoring.SCORING_VERSION + 1
        bad = self.root / "bumped.json"
        bad.write_text(json.dumps(bumped))
        with self.assertRaises(ValueError):
            NeuralTrainer.load(bad, self.snapshot)
        # ② 老检查点（没有该字段）→ 拒绝，而不是默认当成新口径
        legacy = copy.deepcopy(data)
        legacy.pop("scoring_version")
        legacy.pop("scoring_mode")
        old = self.root / "legacy.json"
        old.write_text(json.dumps(legacy))
        with self.assertRaises(ValueError):
            NeuralTrainer.load(old, self.snapshot)
        # ③ 口径一致时正常载入
        self.assertEqual(NeuralTrainer.load(path, self.snapshot).checkpoint(),
                         trainer.checkpoint())

    def test_inconsistent_checkpoint_scoring_mode_rejected(self):
        path, trainer = self.checkpoint_file()
        data = json.loads(path.read_text())
        data["episode_config"]["scoring_mode"] = "platform"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "episode scoring mode"):
            NeuralTrainer.load(path, self.snapshot)

    def test_cli_resume_inherits_platform_scoring_if_unspecified(self):
        import io
        from contextlib import redirect_stdout, redirect_stderr
        trainer = self.trainer(scoring_mode="platform")
        saved = self.root / "platform.json"
        trainer.save(saved)
        common = ["--snapshot", str(self.snapshot), "--resume", str(saved),
                  "--episodes", "0", "--checkpoint", str(self.root / "next.json"),
                  "--report", str(self.root / "report.json")]
        with patch.object(NeuralTrainer, "train", return_value=[]), patch.object(
                NeuralTrainer, "evaluate", return_value={"initial": {"summary": {}}, "trained": {"summary": {}}}), redirect_stdout(io.StringIO()):
            self.assertEqual(main(common), 0)
        data = json.loads((self.root / "next.json").read_text())
        self.assertEqual(data["scoring_mode"], "platform")
        common[common.index("--checkpoint") + 1] = str(self.root / "bad.json")
        common[common.index("--report") + 1] = str(self.root / "bad-report.json")
        with patch.object(NeuralTrainer, "train") as train, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(common + ["--scoring", "verified"])
        train.assert_not_called()

    def test_platform_scoring_mode_is_recorded_and_trains(self):
        trainer = self.trainer(scoring_mode="platform", batch_episodes=2)
        self.assertEqual(trainer.scoring_mode, "platform")
        self.assertEqual(trainer.checkpoint()["scoring_mode"], "platform")
        self.assertEqual(trainer.episode_config["scoring_mode"], "platform")
        self.assertTrue(all(r["status"]["terminated"] for r in trainer.train(2)))
        with self.assertRaises(ValueError):
            self.trainer(scoring_mode="whatever")

    def test_actual_synthetic_fixture_updates_parameters(self):
        trainer = self.trainer(seed=3, batch_episodes=2)
        before = copy.deepcopy(trainer.policy.checkpoint()["params"])
        records = trainer.train(2)
        self.assertTrue(all(r["status"]["terminated"] for r in records))
        self.assertEqual([r["learner_side"] for r in records], list(SIDES))
        self.assertEqual(trainer.updates, 1)
        self.assertNotEqual(before, trainer.policy.checkpoint()["params"])
        for row in records:
            self.assertEqual(row["own_score"], row["status"]["final_scores"][row["learner_side"]])

    def test_only_own_observations_and_raw_terminal_targets(self):
        observed = []
        class ScriptedEpisode:
            def __init__(self, *args, **kwargs):
                self.n = 0
            def reset(self, *, seed):
                self.seed = seed
                return self.status()
            def status(self):
                done = self.n == 4
                return {"current_actor": None if done else SIDES[self.n % 2],
                        "terminated": done, "truncated": False,
                        "final_scores": {"BLUE": 7, "RED": -3} if done else None,
                        "rewards": {"BLUE": 7, "RED": -3} if done else {s: 0 for s in SIDES}}
            def observe(self, side):
                observed.append((self.seed, side))
                return {"side": side, "visible": []}
            def legal_actions(self, side):
                return [{"token": 0}, {"token": 1}]
            def step(self, action, **kwargs):
                self.n += 1
                return self.status()
        with patch("env.neural_train.TwoPlayerEpisode", ScriptedEpisode):
            trainer = self.trainer(batch_episodes=2)
            with patch.object(trainer.policy, "update_batch", wraps=trainer.policy.update_batch) as update:
                trainer.train(2)
                # The list is cleared after update; capture immutable expectations via wrapper below.
                self.assertEqual(update.call_count, 1)
        self.assertEqual(observed, [(0, "BLUE"), (0, "BLUE"), (1, "RED"), (1, "RED")])
        with patch("env.neural_train.TwoPlayerEpisode", ScriptedEpisode):
            trainer = self.trainer(batch_episodes=2)
            batches = []
            original = trainer.policy.update_batch
            def capture(batch):
                batches.append([(len(history), score) for history, score in batch])
                return original(batch)
            trainer.policy.update_batch = capture
            trainer.train(2)
        self.assertEqual(batches, [[(2, 7), (2, -3)]])

    def test_real_intermediate_rewards_are_zero(self):
        episode = TwoPlayerEpisode(self.snapshot, max_seconds=40, max_steps=100)
        status = episode.reset(seed=7)
        saw_intermediate = False
        while not status["terminated"] and not status["truncated"]:
            self.assertEqual(status["rewards"], {s: 0 for s in SIDES})
            saw_intermediate = True
            status = episode.step(episode.legal_actions()[0])
        self.assertTrue(saw_intermediate)
        self.assertEqual(status["rewards"], status["final_scores"])

    def test_truncation_never_updates_or_enters_pending(self):
        trainer = NeuralTrainer(self.snapshot, episode_config={"max_seconds": 999, "max_steps": 2},
                                policy_config={"buckets": 32, "hidden_dim": 4}, batch_episodes=1)
        before = copy.deepcopy(trainer.policy.checkpoint())
        records = trainer.train(2)
        self.assertTrue(all(r["status"]["truncated"] for r in records))
        self.assertEqual(trainer.updates, 0)
        self.assertEqual(trainer.pending, [])
        after = trainer.policy.checkpoint()
        for key in ("params", "m", "v", "steps"):
            self.assertEqual(before[key], after[key])

    def test_resume_matches_uninterrupted_even_mid_batch(self):
        whole = self.trainer(seed=71, batch_episodes=2)
        whole.train(6)
        partial = self.trainer(seed=71, batch_episodes=2)
        partial.train(3)
        path = self.root / "checkpoint.json"
        partial.save(path)
        resumed = NeuralTrainer.load(path, self.snapshot)
        resumed.train(3)
        self.assertEqual(resumed.checkpoint(), whole.checkpoint())
        with self.assertRaises(FileExistsError):
            resumed.save(path)

    def test_frozen_initial_and_trained_paired_evaluation_no_mutation(self):
        trainer = self.trainer(seed=14, batch_episodes=2, heldout_seeds=[90, 91])
        trainer.train(2)
        before = canonical(trainer.checkpoint())
        first, second = trainer.evaluate(), trainer.evaluate()
        for output in (first, second):
            for label in ("initial", "trained"):
                for row in output[label]["games"]:
                    row.pop("elapsed_seconds")
        self.assertEqual(first, second)
        self.assertEqual(canonical(trainer.checkpoint()), before)
        self.assertEqual([(r["seed"], r["learner_side"]) for r in first["initial"]["games"]],
                         [(90, "BLUE"), (90, "RED"), (91, "BLUE"), (91, "RED")])
        with self.assertRaises(ValueError):
            trainer.evaluate([1])
        with self.assertRaises(ValueError):
            trainer.evaluate([90, 90])

    def test_heldout_reservation_blocks_future_training(self):
        trainer = self.trainer(heldout_seeds=[2])
        before = trainer.checkpoint()
        with self.assertRaises(ValueError):
            trainer.train(3)
        self.assertEqual(before, trainer.checkpoint())

    def test_malformed_pending_features_rejected_at_load(self):
        trainer = self.trainer(batch_episodes=2)
        trainer.train(1)
        data = trainer.checkpoint()
        data["pending"][0]["decisions"][0]["observation"][0][0] = 32
        path = self.root / "invalid.json"
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            NeuralTrainer.load(path, self.snapshot)

    def test_snapshot_changes_rejected(self):
        trainer = self.trainer()
        path = self.root / "checkpoint.json"
        trainer.save(path)
        units = self.snapshot / "units.json"
        units.write_text(units.read_text() + "\n")
        with self.assertRaises(ValueError):
            trainer.train(1)
        with self.assertRaises(ValueError):
            NeuralTrainer.load(path, self.snapshot)

    def test_explicit_snapshot_cli_and_resume(self):
        checkpoint, report = self.root / "first.json", self.root / "report.json"
        common = ["--snapshot", str(self.snapshot), "--episodes", "2", "--buckets", "32", "--hidden-dim", "4",
                  "--max-seconds", "10", "--max-steps", "20", "--batch-episodes", "2"]
        self.assertEqual(main(common + ["--checkpoint", str(checkpoint), "--report", str(report)]), 0)
        self.assertEqual(json.loads(report.read_text())["updates"], 1)
        second, report2 = self.root / "second.json", self.root / "report2.json"
        self.assertEqual(main(common + ["--resume", str(checkpoint), "--checkpoint", str(second), "--report", str(report2)]), 0)
        self.assertEqual(json.loads(second.read_text())["episodes_attempted"], 4)


class PublicNeuralTrainingTests(NeuralTrainingTests):
    def trainer(self, **kwargs):
        return super().trainer(observation_schema=2, **kwargs)

    # These inherited tests intentionally exercise a legacy-only fake or width.
    def test_only_own_observations_and_raw_terminal_targets(self):
        from env.public_observation import PublicObservationAdapter
        trainer = self.trainer(batch_episodes=2)
        original_sample = trainer.policy.sample
        original_step = TwoPlayerEpisode.step
        selected, observations, batches = [], [], []
        original_update = trainer.policy.update_batch
        def sample(obs, candidates, **kwargs):
            observations.append(copy.deepcopy(obs))
            result = original_sample(obs, candidates, **kwargs)
            selected.append(result[0]["action"])
            return result
        def step(episode, action, **kwargs):
            self.assertNotIn("public_context", action)
            if selected:
                self.assertIs(action, selected.pop())
            return original_step(episode, action, **kwargs)
        def update(batch):
            batches.extend(score for _, score in batch)
            return original_update(batch)
        with patch.object(trainer.policy, "sample", side_effect=sample), \
                patch.object(trainer.policy, "update_batch", side_effect=update), \
                patch.object(TwoPlayerEpisode, "step", new=step):
            records = trainer.train(2)
        self.assertEqual(batches, [r["status"]["final_scores"][r["learner_side"]] for r in records])
        self.assertEqual([o["side"] for o in observations],
                         [s for s, r in zip(SIDES, records) for _ in range(r["decisions"])])
        for obs in observations:
            self.assertEqual(obs["schema_version"], 2)
            self.assertIn("enemy", obs["score"])
            self.assertIn("board", obs)

    def test_malformed_pending_features_rejected_at_load(self):
        trainer = self.trainer(batch_episodes=2)
        trainer.train(1)
        data = trainer.checkpoint()
        data["pending"][0]["decisions"][0]["observation"][0][0] = 64
        path = self.root / "invalid.json"
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            NeuralTrainer.load(path, self.snapshot)

    def test_truncation_never_updates_or_enters_pending(self):
        trainer = NeuralTrainer(self.snapshot, observation_schema=2,
                                episode_config={"max_seconds": 999, "max_steps": 2},
                                policy_config={"buckets": 32, "hidden_dim": 4}, batch_episodes=1)
        before = copy.deepcopy(trainer.policy.checkpoint())
        records = trainer.train(2)
        self.assertEqual(trainer.observation_schema, 2)
        self.assertEqual(trainer.policy.config["representation"], "public_v2_split")
        for row in records:
            self.assertTrue(row["status"]["truncated"])
            self.assertFalse(row["status"]["terminated"])
            self.assertIsNone(row["status"]["final_scores"])
            self.assertIsNone(row["own_score"])
            self.assertEqual(row["status"]["rewards"], {s: 0 for s in SIDES})
        self.assertEqual(trainer.updates, 0)
        self.assertEqual(trainer.pending, [])
        after = trainer.policy.checkpoint()
        for key in ("params", "m", "v", "steps"):
            self.assertEqual(before[key], after[key])

    def test_schema_explicit_and_mismatch_rejected(self):
        trainer = self.trainer()
        self.assertEqual(trainer.checkpoint()["observation_schema"], 2)
        self.assertEqual(trainer.policy.config["representation"], "public_v2_split")
        for schema in (True, 0, 5, 2.0):
            with self.assertRaises(ValueError):
                NeuralTrainer(self.snapshot, observation_schema=schema)
        with self.assertRaises(ValueError):
            NeuralTrainer(self.snapshot, observation_schema=2,
                          policy_config={"representation": "legacy_json"})
        path = self.root / "schema.json"
        trainer.save(path)
        data = json.loads(path.read_text())
        data.pop("observation_schema")
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            NeuralTrainer.load(path, self.snapshot)

    def test_public_cli_resume_inherits_schema_and_rejects_override(self):
        path, report = self.root / "v2.json", self.root / "v2-report.json"
        args = ["--snapshot", str(self.snapshot), "--episodes", "1", "--buckets", "16",
                "--hidden-dim", "4", "--max-seconds", "10", "--max-steps", "20"]
        main(args + ["--observation-schema", "2", "--checkpoint", str(path), "--report", str(report)])
        self.assertEqual(json.loads(report.read_text())["observation_schema"], 2)
        resumed, report2 = self.root / "resumed.json", self.root / "resumed-report.json"
        main(args + ["--resume", str(path), "--checkpoint", str(resumed), "--report", str(report2)])
        self.assertEqual(json.loads(resumed.read_text())["observation_schema"], 2)
        bad, bad_report = self.root / "bad.json", self.root / "bad-report.json"
        with self.assertRaises(SystemExit):
            main(args + ["--resume", str(path), "--observation-schema", "1",
                         "--checkpoint", str(bad), "--report", str(bad_report)])
        self.assertFalse(bad.exists())
        self.assertFalse(bad_report.exists())

    def test_hidden_changes_leave_encoded_policy_input_unchanged(self):
        from engine.game import Game
        from env.public_observation import PublicObservationAdapter
        trainer = self.trainer()
        with patch.object(Game, "can_observe", return_value=False):
            episode = TwoPlayerEpisode(self.snapshot)
            episode.reset(seed=1)
            adapter = PublicObservationAdapter(episode)
            before = trainer.policy._decision(adapter.observe("BLUE"), adapter.candidate_views("BLUE"))
            hidden = episode._session.env.game.units["red"]
            hidden.pos = 20008
            # 不扰动 `blood`：它已通过公开的「剩余兵力分」合计反出（平台行为），
            # 见 tests/test_public_observation.py::
            #     test_public_score_reveals_only_aggregate_enemy_strength
            hidden.ammo["a5"] = 999
            hidden.detail["target"] = "secret"
            hidden.stunned_until = 12345
            after = trainer.policy._decision(adapter.observe("BLUE"), adapter.candidate_views("BLUE"))
        self.assertEqual(before.to_dict(), after.to_dict())

    def test_legacy_checkpoint_format_still_loads_without_new_config(self):
        trainer = NeuralTrainer(self.snapshot, policy_config={"buckets": 16, "hidden_dim": 4},
                                episode_config={"max_seconds": 10, "max_steps": 20})
        trainer.train(1)
        data = trainer.checkpoint()
        self.assertNotIn("observation_schema", data)
        self.assertNotIn("representation", data["policy"]["config"])
        path = self.root / "old.json"
        trainer.save(path)
        restored = NeuralTrainer.load(path, self.snapshot)
        self.assertEqual(restored.checkpoint(), data)
        self.assertEqual(restored.observation_schema, 1)


if __name__ == "__main__":
    unittest.main()
