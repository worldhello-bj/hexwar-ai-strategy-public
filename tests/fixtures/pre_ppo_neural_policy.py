# Frozen compatibility reference, not production code.
# Source: env/neural_policy.py at 1c9bfad061aede301b64a624811568e344e33207
# Git blob: f451d92699ac50b2ee7bc007f05b42327d8176b7
# The source below is unchanged. Keep independent of env.neural_policy so
# compatibility tests compare old/new arithmetic within the same runtime.
"""Generic JSON policy gradient learner; no engine imports or tactical features.

Each completed trajectory uses its own original terminal score at every decision
(gamma=1). Advantages use pre-update, detached observation-only predictions.
Feature hashing and signed-log numeric encoding are domain independent.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class Decision:
    observation: tuple
    candidates: tuple
    selected: int

    def to_dict(self):
        def pair(value):
            return [value[0].tolist(), value[1].tolist()]
        return {"observation": pair(self.observation),
                "candidates": [pair(value) for value in self.candidates],
                "selected": self.selected}

    @classmethod
    def from_dict(cls, data):
        def pair(value):
            indices, values = np.asarray(value[0], dtype=np.int64), np.asarray(value[1], dtype=np.float64)
            if indices.ndim != 1 or values.shape != indices.shape or not np.isfinite(values).all() or (indices < 0).any():
                raise ValueError("invalid sparse decision features")
            return indices, values
        selected = data["selected"]
        candidates = tuple(pair(value) for value in data["candidates"])
        if isinstance(selected, bool) or not isinstance(selected, int) or not 0 <= selected < len(candidates):
            raise ValueError("invalid selected candidate")
        return cls(pair(data["observation"]), candidates, selected)


class NeuralPolicy:
    def __init__(self, *, seed=0, buckets=2048, hidden_dim=32,
                 learning_rate=0.001, gradient_clip=5.0, hash_seed=0, representation="legacy_json",
                 baseline_offset="none"):
        for name, value in (("buckets", buckets), ("hidden_dim", hidden_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(name + " must be a positive integer")
        for name, value in (("learning_rate", learning_rate), ("gradient_clip", gradient_clip)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        self.config = dict(seed=seed, buckets=buckets, hidden_dim=hidden_dim,
                           learning_rate=learning_rate, gradient_clip=gradient_clip,
                           hash_seed=hash_seed)
        if representation not in ("legacy_json", "public_v2_split"):
            raise ValueError("unsupported policy representation")
        # Omit the legacy default to preserve the original checkpoint format.
        if representation != "legacy_json":
            self.config["representation"] = representation
        if baseline_offset not in ("none", "prior_batch_mean"):
            raise ValueError("unsupported baseline offset")
        if baseline_offset != "none":
            self.config["baseline_offset"] = baseline_offset
        self.return_mean = 0.0
        self.return_count = 0
        self.feature_width = buckets * (2 if representation == "public_v2_split" else 1)
        self.rng = np.random.default_rng(seed)
        self.params = {"observation": self.rng.normal(0, 0.05, (self.feature_width, hidden_dim)),
                       "action": self.rng.normal(0, 0.05, (self.feature_width, hidden_dim)),
                       "observation_bias": np.zeros(hidden_dim),
                       "action_bias": np.zeros(hidden_dim),
                       "policy": self.rng.normal(0, 0.05, 3 * hidden_dim),
                       "value": self.rng.normal(0, 0.05, hidden_dim),
                       "value_bias": np.zeros(1)}
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.steps = 0

    def _features(self, value):
        accumulated = {}
        def emit(path, kind, token=None, amount=1.0):
            identity = json.dumps([self.config["hash_seed"], path, kind, token],
                                  ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            digest = hashlib.blake2b(identity.encode(), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "little") % self.config["buckets"]
            sign = 1.0 if digest[8] & 1 else -1.0
            accumulated[index] = accumulated.get(index, 0.0) + sign * amount
        def visit(item, path):
            if item is None:
                emit(path, "null")
            elif isinstance(item, bool):
                emit(path, "bool", item)
            elif isinstance(item, (int, float)):
                if not math.isfinite(item):
                    raise ValueError("JSON numeric features must be finite")
                emit(path, "number_present")
                emit(path, "number", amount=math.copysign(math.log1p(abs(item)), item))
            elif isinstance(item, str):
                emit(path, "string", item)
            elif isinstance(item, dict):
                if not all(isinstance(k, str) for k in item):
                    raise ValueError("JSON object keys must be strings")
                emit(path, "object")
                for key in sorted(item):
                    visit(item[key], path + [["key", key]])
            elif isinstance(item, list):
                emit(path, "array")
                for index, child in enumerate(item):
                    visit(child, path + [["index", index]])
            else:
                raise ValueError("features require JSON values")
        visit(value, [])
        # Generic input norm controls saturation without altering returns.
        keys = np.array(sorted(accumulated), dtype=np.int64)
        values = np.array([accumulated[k] for k in keys], dtype=np.float64)
        norm = np.linalg.norm(values)
        if norm > 0:
            values /= norm
        return keys, values

    def encode(self, value):
        indices, values = self._features(value)
        result = np.zeros(self.config["buckets"])
        result[indices] = values
        return result

    @staticmethod
    def _project(features, weight, bias):
        # One batched sparse projection for all candidate actions.
        sizes = [len(indices) for indices, _ in features]
        rows = np.repeat(np.arange(len(features)), sizes)
        indices = np.concatenate([x[0] for x in features])
        values = np.concatenate([x[1] for x in features])
        result = np.broadcast_to(bias, (len(features), len(bias))).copy()
        np.add.at(result, rows, weight[indices] * values[:, None])
        return np.tanh(result)

    def _forward(self, decision):
        p = self.params
        obs = self._project((decision.observation,), p["observation"], p["observation_bias"])[0]
        actions = self._project(decision.candidates, p["action"], p["action_bias"])
        joint = np.concatenate((np.broadcast_to(obs, actions.shape), actions, actions * obs), axis=1)
        logits = joint @ p["policy"]
        shifted = logits - np.max(logits)
        log_probs = shifted - np.log(np.exp(shifted).sum())
        probs = np.exp(log_probs)
        value = float(obs @ p["value"] + p["value_bias"][0])
        return obs, actions, joint, probs, value, log_probs

    def _split_features(self, first, second, *, cache_second=False):
        """Separate learned banks with independent unit norms, no joint norm.

        A board of any size therefore cannot change or collide with the dynamic
        features. The same generic operation separates action from its supplied
        public context. It introduces no distance, target or tactical features.
        """
        left = self._features(first)
        if cache_second:
            # Cache only a canonical immutable key and derived features. Input
            # mutation invalidates the cache; no engine state or RNG is cached.
            key = json.dumps(second, sort_keys=True, separators=(",", ":"), allow_nan=False)
            if key != getattr(self, "_board_cache_key", None):
                self._board_cache_features = self._features(second)
                self._board_cache_key = key
            right = self._board_cache_features
        else:
            right = self._features(second)
        return (np.concatenate((left[0], right[0] + self.config["buckets"])),
                np.concatenate((left[1], right[1])))

    def _decision(self, observation, candidates, selected=0):
        if not candidates:
            raise ValueError("policy needs at least one supplied candidate")
        if self.config.get("representation", "legacy_json") == "public_v2_split":
            if not isinstance(observation, dict) or observation.get("schema_version") != 2 or "board" not in observation:
                raise ValueError("public_v2_split requires a schema-v2 public observation")
            if any(not isinstance(a, dict) or set(a) != {"action", "public_context"} for a in candidates):
                raise ValueError("public_v2_split requires public candidate views")
            dynamic = {k: v for k, v in observation.items() if k != "board"}
            return Decision(self._split_features(dynamic, observation["board"], cache_second=True),
                            tuple(self._split_features(a["action"], a["public_context"]) for a in candidates), selected)
        return Decision(self._features(observation), tuple(self._features(a) for a in candidates), selected)

    def probabilities(self, observation, candidates):
        return self._forward(self._decision(observation, candidates))[3]

    def sample(self, observation, candidates):
        record = self._decision(observation, candidates)
        probs = self._forward(record)[3]
        index = int(self.rng.choice(len(candidates), p=probs))
        return candidates[index], Decision(record.observation, record.candidates, index)

    def select(self, observation, candidates):
        return self.sample(observation, candidates)[0]

    __call__ = select

    def _loss_gradients(self, examples, advantages=None, actor_normalizer=None, baseline_offset=0.0):
        """Actor sum / actor_normalizer + mean half squared value error.

        Explicit advantages are for derivative checks; update computes them once
        from the pre-update parameters, never normalizes/clips rewards.
        """
        gradients = {k: np.zeros_like(v) for k, v in self.params.items()}
        if not examples:
            return 0.0, gradients
        count = len(examples)
        if actor_normalizer is None:
            actor_normalizer = count
        if actor_normalizer <= 0:
            raise ValueError("actor normalizer must be positive")
        # The offset is fixed from prior completed batches, never this batch.
        # These residual targets are internal critic targets, not changed returns.
        if baseline_offset != 0.0:
            examples = [(d, score - baseline_offset) for d, score in examples]
        forward = [self._forward(d) for d, _ in examples]
        if advantages is None:
            advantages = [score - f[4] for (_, score), f in zip(examples, forward)]
        loss = 0.0
        h = self.config["hidden_dim"]
        for (decision, score), adv, f in zip(examples, advantages, forward):
            obs, actions, joint, probs, value, log_probs = f
            loss += -adv * log_probs[decision.selected] / actor_normalizer + 0.5 * (value - score) ** 2 / count
            dlogits = probs.copy()
            dlogits[decision.selected] -= 1
            dlogits *= adv / actor_normalizer
            gradients["policy"] += joint.T @ dlogits
            w = self.params["policy"]
            dobs = (dlogits[:, None] * (w[:h] + actions * w[2*h:])).sum(axis=0)
            dactions = dlogits[:, None] * (w[h:2*h] + obs * w[2*h:])
            error = (value - score) / count
            gradients["value"] += error * obs
            gradients["value_bias"][0] += error
            dobs += error * self.params["value"]
            dobs *= 1 - obs ** 2
            dactions *= 1 - actions ** 2
            indices, values = decision.observation
            np.add.at(gradients["observation"], indices, values[:, None] * dobs)
            gradients["observation_bias"] += dobs
            gradients["action_bias"] += dactions.sum(axis=0)
            for (indices, values), derivative in zip(decision.candidates, dactions):
                np.add.at(gradients["action"], indices, values[:, None] * derivative)
        return float(loss), gradients

    def update_batch(self, trajectories):
        """Update from [(completed own-seat decisions, raw own terminal score)].

        The caller must exclude truncated/unfinished trajectories. No intermediate
        rewards, discount, entropy bonus, return scaling or tactical signal.
        Actor terms sum within each trajectory then average over all completed
        trajectories supplied (including empty ones with zero actor contribution).
        Value regression remains a mean over decisions.
        """
        examples = []
        trajectory_count = 0
        scores = []
        for history, score in trajectories:
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("terminal score must be finite")
            trajectory_count += 1
            scores.append(float(score))
            for decision in history:
                if not isinstance(decision, Decision) or not 0 <= decision.selected < len(decision.candidates):
                    raise ValueError("invalid decision")
                examples.append((decision, float(score)))
        use_offset = self.config.get("baseline_offset", "none") == "prior_batch_mean"
        offset = self.return_mean if use_offset else 0.0
        # Stage statistics without exposing current returns to the loss. Iterating
        # in trajectory order makes statistics independent of batch partitioning.
        next_mean, next_count = self.return_mean, self.return_count
        if use_offset:
            for score in scores:
                next_count += 1
                # Weighted form avoids overflow in (score - mean).
                next_mean = next_mean * ((next_count - 1) / next_count) + score / next_count
            if not math.isfinite(next_mean):
                raise FloatingPointError("non-finite running mean; parameters unchanged")
        loss, gradients = self._loss_gradients(examples, actor_normalizer=trajectory_count,
                                              baseline_offset=offset)
        if not examples:
            # A completed no-decision episode counts once; an empty batch does
            # nothing. There is no optimizer step when there are no decisions.
            self.return_mean, self.return_count = next_mean, next_count
            return {"decisions": 0, "loss": loss, "gradient_norm": 0.0}
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients.values()))
        if not math.isfinite(loss) or not math.isfinite(norm):
            raise FloatingPointError("non-finite loss or gradient; parameters unchanged")
        scale = min(1.0, self.config["gradient_clip"] / max(norm, 1e-30))
        self.steps += 1
        for key, gradient in gradients.items():
            gradient *= scale
            self.m[key] = 0.9 * self.m[key] + 0.1 * gradient
            self.v[key] = 0.999 * self.v[key] + 0.001 * gradient ** 2
            mhat = self.m[key] / (1 - 0.9 ** self.steps)
            vhat = self.v[key] / (1 - 0.999 ** self.steps)
            self.params[key] -= self.config["learning_rate"] * mhat / (np.sqrt(vhat) + 1e-8)
        self.return_mean, self.return_count = next_mean, next_count
        return {"decisions": len(examples), "loss": loss, "gradient_norm": norm}

    def checkpoint(self):
        return {"schema_version": 1, "algorithm": "generic_json_reinforce",
                "config": dict(self.config), "steps": self.steps,
                "rng": self.rng.bit_generator.state,
                **({"return_baseline": {"mean": self.return_mean, "count": self.return_count}}
                   if self.config.get("baseline_offset") == "prior_batch_mean" else {}),
                **{name: {k: v.tolist() for k, v in values.items()}
                   for name, values in (("params", self.params), ("m", self.m), ("v", self.v))}}

    @classmethod
    def from_checkpoint(cls, data):
        if data.get("schema_version") != 1 or data.get("algorithm") != "generic_json_reinforce":
            raise ValueError("unsupported checkpoint")
        result = cls(**data["config"])
        if result.config.get("baseline_offset") == "prior_batch_mean":
            state = data.get("return_baseline")
            if not isinstance(state, dict) or set(state) != {"mean", "count"}:
                raise ValueError("missing or invalid return baseline state")
            mean, count = state["mean"], state["count"]
            if (isinstance(count, bool) or not isinstance(count, int) or count < 0 or
                    isinstance(mean, bool) or not isinstance(mean, (int, float)) or
                    not math.isfinite(mean) or (count == 0 and mean != 0)):
                raise ValueError("invalid return baseline statistics")
            result.return_mean, result.return_count = float(mean), count
        elif "return_baseline" in data:
            raise ValueError("return baseline state requires explicit configuration")
        for name in ("params", "m", "v"):
            target = getattr(result, name)
            if set(data[name]) != set(target):
                raise ValueError("checkpoint parameter keys differ")
            for key in target:
                array = np.array(data[name][key], dtype=np.float64)
                if array.shape != target[key].shape or not np.isfinite(array).all():
                    raise ValueError("invalid checkpoint tensor")
                target[key] = array
        if isinstance(data["steps"], bool) or not isinstance(data["steps"], int) or data["steps"] < 0:
            raise ValueError("invalid optimizer step")
        result.steps = data["steps"]
        result.rng.bit_generator.state = data["rng"]
        return result

    def save(self, path):
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(self.checkpoint(), stream, allow_nan=False, separators=(",", ":"))
            stream.write("\n")

    @classmethod
    def load(cls, path):
        return cls.from_checkpoint(json.loads(Path(path).read_text("utf-8")))
