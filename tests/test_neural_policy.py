import ast
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from env.neural_policy import Decision, NeuralPolicy


class NeuralPolicyTests(unittest.TestCase):
    def model(self, **kwargs):
        return NeuralPolicy(buckets=64, hidden_dim=3, **kwargs)

    def test_canonical_dict_order_and_array_order(self):
        p = self.model()
        np.testing.assert_array_equal(p.encode({"a": 1, "b": [2, 3]}), p.encode({"b": [2, 3], "a": 1}))
        self.assertFalse(np.array_equal(p.encode([2, 3]), p.encode([3, 2])))
        self.assertFalse(np.array_equal(p.encode(True), p.encode(1)))
        self.assertFalse(np.array_equal(p.encode("1"), p.encode(1)))
        self.assertFalse(np.array_equal(p.encode({"0": 1}), p.encode([1])))
        for invalid in [float("nan"), float("inf"), {1: "x"}, object()]:
            with self.assertRaises(ValueError):
                p.encode(invalid)

    def test_probabilities_order_and_legal_identity(self):
        p = self.model()
        actions = [{"x": 1}, {"x": 2}, {"x": -3}]
        probabilities = p.probabilities({"state": 5}, actions)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue((probabilities > 0).all())
        self.assertAlmostEqual(sum(probabilities), 1)
        np.testing.assert_allclose(probabilities[[2, 0, 1]], p.probabilities({"state": 5}, [actions[i] for i in [2, 0, 1]]), atol=1e-14)
        for _ in range(20):
            action, decision = p.sample({}, actions)
            self.assertIs(action, actions[decision.selected])
        with self.assertRaises(ValueError):
            p.select({}, [])
        self.assertIs(p.select({}, [actions[0]]), actions[0])

    def test_sample_can_return_probabilities_without_duplicate_forward(self):
        from unittest.mock import patch
        p, q = self.model(seed=51), self.model(seed=51)
        obs, candidates = {"s": 2}, [{"a": 1}, {"a": 2}]
        expected_probs = q.probabilities(obs, candidates)
        expected_action, expected_decision = q.sample(obs, candidates)
        with patch.object(p, "_forward", wraps=p._forward) as forward:
            action, decision, probs = p.sample(obs, candidates, return_probabilities=True)
        self.assertEqual(forward.call_count, 1)
        self.assertEqual(action, expected_action)
        self.assertEqual(decision.to_dict(), expected_decision.to_dict())
        np.testing.assert_array_equal(probs, expected_probs)
        self.assertEqual(p.checkpoint(), q.checkpoint())

    def test_observation_action_product_changes_preferences(self):
        p = self.model()
        actions = [{"x": 1}, {"x": 2}]
        first = p.probabilities({"state": 1}, actions)
        second = p.probabilities({"state": -1}, actions)
        self.assertGreater(np.max(abs(first - second)), 1e-10)
        p.params["policy"][6:] = 0
        np.testing.assert_allclose(p.probabilities({"state": 1}, actions), p.probabilities({"state": -1}, actions), atol=1e-14)

    def test_every_parameter_gradient_by_finite_difference(self):
        p = NeuralPolicy(buckets=8, hidden_dim=2)
        examples = [(p._decision({"a": 2}, [{"b": 1}, {"b": -2}], 1), 3.5),
                    (p._decision([None, False], ["a", "b", "c"], 0), -1.2)]
        advantages = [score - p._forward(d)[4] for d, score in examples]
        _, gradients = p._loss_gradients(examples, advantages)
        eps = 1e-5
        for name, parameter in p.params.items():
            for index in np.ndindex(parameter.shape):
                old = parameter[index]
                parameter[index] = old + eps
                plus = p._loss_gradients(examples, advantages)[0]
                parameter[index] = old - eps
                minus = p._loss_gradients(examples, advantages)[0]
                parameter[index] = old
                self.assertAlmostEqual(gradients[name][index], (plus - minus) / (2 * eps), places=7, msg=str((name, index)))

    def test_variable_length_actor_is_episode_mean(self):
        p = self.model()
        short = p._decision({"s": 1}, [1, 2], 0)
        long = p._decision({"s": 2}, [1, 2], 1)
        examples = [(short, 3.0), (long, 7.0), (long, 7.0), (long, 7.0)]
        advantages = [score - p._forward(d)[4] for d, score in examples]
        _, episode_gradients = p._loss_gradients(examples, advantages, actor_normalizer=2)
        _, decision_gradients = p._loss_gradients(examples, advantages)
        np.testing.assert_allclose(episode_gradients["policy"], 2 * decision_gradients["policy"])
        np.testing.assert_allclose(episode_gradients["value"], decision_gradients["value"])
        eps = 1e-5
        for name, parameter in p.params.items():
            index = (0,) * parameter.ndim
            old = parameter[index]
            parameter[index] = old + eps
            plus = p._loss_gradients(examples, advantages, actor_normalizer=2)[0]
            parameter[index] = old - eps
            minus = p._loss_gradients(examples, advantages, actor_normalizer=2)[0]
            parameter[index] = old
            self.assertAlmostEqual(episode_gradients[name][index], (plus - minus) / (2 * eps), places=7)
        # Capture the actual update's normalization, including an empty episode.
        from unittest.mock import patch
        with patch.object(p, "_loss_gradients", wraps=p._loss_gradients) as compute:
            p.update_batch([([short], 3.0), ([long, long, long], 7.0), ([], 0.0)])
            self.assertEqual(compute.call_args.kwargs["actor_normalizer"], 3)

    def test_checkpoint_resume_exact_including_optimizer_rng(self):
        p = self.model(seed=10)
        actions = [{"a": 1}, {"a": 2}]
        _, record = p.sample({"s": 7}, actions)
        p.update_batch([([record], 170.0)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            p.save(path)
            q = NeuralPolicy.load(path)
            with self.assertRaises(FileExistsError):
                p.save(path)
        saved_record = Decision.from_dict(json.loads(json.dumps(record.to_dict())))
        p.update_batch([([record], -7)])
        q.update_batch([([saved_record], -7)])
        for _ in range(20):
            self.assertEqual(p.sample({}, actions)[1].selected, q.sample({}, actions)[1].selected)
        self.assertEqual(p.checkpoint(), q.checkpoint())

    def test_raw_return_empty_batch_and_baseline(self):
        p = self.model()
        d = p._decision({}, [1, 2])
        predicted = p._forward(d)[4]
        loss, gradients = p._loss_gradients([(d, 123.0)])
        self.assertAlmostEqual(gradients["value_bias"][0], predicted - 123.0)
        self.assertGreater(loss, 1000)
        before = p.checkpoint()
        self.assertEqual(p.update_batch([])["decisions"], 0)
        self.assertEqual(before, p.checkpoint())
        with self.assertRaises(ValueError):
            p.update_batch([([d], float("nan"))])
        self.assertEqual(before, p.checkpoint())
        report = p.update_batch([([d], 123.0)])
        self.assertEqual(report["decisions"], 1)
        self.assertEqual(p.steps, 1)

    def test_no_private_environment_import_or_access(self):
        import env.neural_policy as module
        source = inspect.getsource(module)
        tree = ast.parse(source)
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [alias.name for n in ast.walk(tree) if isinstance(n, ast.Import) for alias in n.names]
        self.assertEqual(set(imports), {"dataclasses", "hashlib", "json", "math", "pathlib", "numpy", "scipy.sparse", "compact_checkpoint", "training_diagnostics"})
        import env.training_diagnostics as diagnostics
        diagnostics_tree = ast.parse(inspect.getsource(diagnostics))
        diagnostics_imports = [n.module for n in ast.walk(diagnostics_tree) if isinstance(n, ast.ImportFrom)]
        diagnostics_imports += [a.name for n in ast.walk(diagnostics_tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertEqual(set(diagnostics_imports), {"math", "numpy"})
        import env.compact_checkpoint as transport
        transport_tree = ast.parse(inspect.getsource(transport))
        transport_imports = [n.module for n in ast.walk(transport_tree) if isinstance(n, ast.ImportFrom)]
        transport_imports += [a.name for n in ast.walk(transport_tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertEqual(set(transport_imports), {"json", "math", "pathlib", "zipfile", "numpy", "hashlib"})
        self.assertNotIn("getattr(env", source)


class SplitPublicPolicyTests(unittest.TestCase):
    def fixture(self):
        policy = NeuralPolicy(buckets=64, hidden_dim=3, representation="public_v2_split")
        obs = {"schema_version": 2, "score": {"own": 3, "enemy": 7},
               "time_left": 100, "board": {"cells": {"1": {"elevation": 2}}}}
        candidates = [{"action": {"type": "move", "unit": "a", "path": [1, 2]},
                       "public_context": {"path": [{"elevation": 1}, {"elevation": 2}]}},
                      {"action": {"type": "wait"}, "public_context": {}}]
        return policy, obs, candidates

    def test_independent_banks_preserve_dynamic_features_with_320_cells(self):
        import copy
        policy, obs, candidates = self.fixture()
        first = policy._decision(obs, candidates)
        bigger = copy.deepcopy(obs)
        bigger["board"]["cells"] = {str(i): {"elevation": i, "terrain_code": "8"} for i in range(320)}
        second = policy._decision(bigger, candidates)
        def bank(features, upper):
            indices, values = features
            keep = indices >= 64 if upper else indices < 64
            return indices[keep], values[keep]
        for a, b in zip(bank(first.observation, False), bank(second.observation, False)):
            np.testing.assert_array_equal(a, b)
        self.assertAlmostEqual(np.linalg.norm(bank(second.observation, False)[1]), 1)
        self.assertAlmostEqual(np.linalg.norm(bank(second.observation, True)[1]), 1)
        self.assertGreater(np.max(abs(policy.probabilities(obs, candidates) -
                                     policy.probabilities(bigger, candidates))), 1e-10)
        changed_score = copy.deepcopy(bigger)
        changed_score["score"]["enemy"] = 8
        third = policy._decision(changed_score, candidates)
        self.assertFalse(np.array_equal(bank(second.observation, False)[1], bank(third.observation, False)[1]))
        for a, b in zip(bank(second.observation, True), bank(third.observation, True)):
            np.testing.assert_array_equal(a, b)

    def test_board_cache_reuses_only_unchanged_public_content(self):
        from unittest.mock import patch
        policy, obs, candidates = self.fixture()
        with patch.object(policy, "_features", wraps=policy._features) as features:
            policy.probabilities(obs, candidates)
            first_calls = features.call_count
            policy.probabilities(obs, candidates)
            self.assertEqual(features.call_count - first_calls, first_calls - 1)
            previous = features.call_count
            obs["board"]["cells"]["1"]["elevation"] = 9
            changed = policy.probabilities(obs, candidates)
            self.assertEqual(features.call_count - previous, first_calls)
        restored = NeuralPolicy.from_checkpoint(policy.checkpoint())
        np.testing.assert_array_equal(changed, restored.probabilities(obs, candidates))

    def test_candidate_action_and_context_independent_and_original_identity(self):
        import copy
        policy, obs, candidates = self.fixture()
        first = policy._decision(obs, candidates).candidates[0]
        changed = copy.deepcopy(candidates)
        changed[0]["public_context"] = {"path": [{"elevation": i} for i in range(320)]}
        second = policy._decision(obs, changed).candidates[0]
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a[first[0] < 64], b[second[0] < 64])
        choice, record = policy.sample(obs, candidates)
        self.assertIs(choice, candidates[record.selected])
        with self.assertRaises(ValueError):
            policy.probabilities({}, candidates)
        with self.assertRaises(ValueError):
            policy.probabilities(obs, [c["action"] for c in candidates])

    def test_split_banks_gradients_and_checkpoint_roundtrip(self):
        policy, obs, candidates = self.fixture()
        decision = policy._decision(obs, candidates)
        examples = [(decision, 5)]
        advantages = [5 - policy._forward(decision)[4]]
        _, gradients = policy._loss_gradients(examples, advantages)
        # Both map/dynamic and action/context banks receive correct derivatives.
        for name, feature in (("observation", decision.observation), ("action", decision.candidates[0])):
            for bank in (False, True):
                index = int(feature[0][(feature[0] >= 64) == bank][0])
                old = policy.params[name][index, 0]
                policy.params[name][index, 0] = old + 1e-5
                plus = policy._loss_gradients(examples, advantages)[0]
                policy.params[name][index, 0] = old - 1e-5
                minus = policy._loss_gradients(examples, advantages)[0]
                policy.params[name][index, 0] = old
                self.assertAlmostEqual(gradients[name][index, 0], (plus - minus) / 2e-5, places=7)
        policy.update_batch([([decision], 5)])
        restored = NeuralPolicy.from_checkpoint(policy.checkpoint())
        self.assertEqual(restored.checkpoint(), policy.checkpoint())
        np.testing.assert_array_equal(policy.probabilities(obs, candidates), restored.probabilities(obs, candidates))


if __name__ == "__main__":
    unittest.main()
