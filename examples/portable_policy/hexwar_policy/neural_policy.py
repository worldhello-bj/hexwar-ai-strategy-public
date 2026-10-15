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
    # None denotes legacy decisions; zero is a valid explicitly observed score.
    public_score_offset: float | None = None

    def to_dict(self, *, arrays=False):
        def pair(value):
            return list(value) if arrays else [value[0].tolist(), value[1].tolist()]
        return {"observation": pair(self.observation),
                "candidates": [pair(value) for value in self.candidates],
                "selected": self.selected,
                **({"public_score_offset": self.public_score_offset}
                   if self.public_score_offset is not None else {})}

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
        offset = data.get("public_score_offset")
        if "public_score_offset" in data and (isinstance(offset, bool) or
                not isinstance(offset, (int, float)) or not math.isfinite(offset)):
            raise ValueError("invalid public score offset")
        return cls(pair(data["observation"]), candidates, selected, offset)


class NeuralPolicy:
    def __init__(self, *, seed=0, buckets=2048, hidden_dim=32,
                 learning_rate=0.001, gradient_clip=5.0, hash_seed=0, representation="legacy_json",
                 baseline_offset="none", update_mode="reinforce", ppo_clip=None, update_epochs=None,
                 projection_backend="numpy"):
        for name, value in (("buckets", buckets), ("hidden_dim", hidden_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(name + " must be a positive integer")
        for name, value in (("learning_rate", learning_rate), ("gradient_clip", gradient_clip)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        self.config = dict(seed=seed, buckets=buckets, hidden_dim=hidden_dim,
                           learning_rate=learning_rate, gradient_clip=gradient_clip,
                           hash_seed=hash_seed)
        if representation not in ("legacy_json", "public_v2_split", "public_v3_spatial", "public_v4_contacts"):
            raise ValueError("unsupported policy representation")
        # Omit the legacy default to preserve the original checkpoint format.
        if representation != "legacy_json":
            self.config["representation"] = representation
        if baseline_offset not in ("none", "prior_batch_mean", "public_own_score"):
            raise ValueError("unsupported baseline offset")
        if baseline_offset != "none":
            self.config["baseline_offset"] = baseline_offset
        if update_mode not in ("reinforce", "ppo_clip"):
            raise ValueError("unsupported update mode")
        if update_mode == "reinforce":
            if ppo_clip is not None or update_epochs is not None:
                raise ValueError("PPO settings require update_mode=ppo_clip")
        else:
            ppo_clip = 0.2 if ppo_clip is None else ppo_clip
            update_epochs = 4 if update_epochs is None else update_epochs
            if isinstance(ppo_clip, bool) or not math.isfinite(ppo_clip) or not 0 < ppo_clip < 1:
                raise ValueError("ppo_clip must be finite and between zero and one")
            if isinstance(update_epochs, bool) or not isinstance(update_epochs, int) or not 1 <= update_epochs <= 100:
                raise ValueError("update_epochs must be an integer between 1 and 100")
            self.config.update(update_mode=update_mode, ppo_clip=ppo_clip, update_epochs=update_epochs)
        if projection_backend not in ("numpy", "scipy_csr"):
            raise ValueError("unsupported projection backend")
        if projection_backend == "scipy_csr":
            from scipy.sparse import csr_matrix
            self._csr_matrix = csr_matrix
            self.config["projection_backend"] = projection_backend
        self.return_mean = 0.0
        self.return_count = 0
        self.feature_width = buckets * (2 if representation in ("public_v2_split", "public_v3_spatial", "public_v4_contacts") else 1)
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
        # Only hash identities, never feature amounts or mutable observations.
        # Tuple paths serialize byte-for-byte like the previous list paths.
        config_key = json.dumps([self.config["hash_seed"], self.config["buckets"]],
                                ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        if config_key != getattr(self, "_feature_hash_config", None):
            self._feature_hash_cache = {}
            self._feature_hash_config = config_key
        cache = self._feature_hash_cache
        def emit(path, kind, token=None, amount=1.0):
            key = (path, kind, token)
            cached = cache.get(key)
            if cached is None:
                identity = json.dumps([self.config["hash_seed"], path, kind, token],
                                      ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                digest = hashlib.blake2b(identity.encode(), digest_size=16).digest()
                index = int.from_bytes(digest[:8], "little") % self.config["buckets"]
                sign = 1.0 if digest[8] & 1 else -1.0
                if len(cache) >= 32768:
                    cache.clear()
                cache[key] = (index, sign)
            else:
                index, sign = cached
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
                    visit(item[key], path + (("key", key),))
            elif isinstance(item, list):
                emit(path, "array")
                for index, child in enumerate(item):
                    visit(child, path + (("index", index),))
            else:
                raise ValueError("features require JSON values")
        visit(value, ())
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

    def _sparse_features(self, features, width):
        sizes = [len(indices) for indices, _ in features]
        indptr = np.concatenate(([0], np.cumsum(sizes, dtype=np.int64)))
        indices = np.concatenate([x[0] for x in features])
        values = np.concatenate([x[1] for x in features])
        return self._csr_matrix((values, indices, indptr), shape=(len(features), width))

    def _project(self, features, weight, bias):
        if self.config.get("projection_backend") == "scipy_csr":
            # No nnz-by-hidden gather/product temporary. Bias addition changes
            # floating accumulation order slightly; the backend is opt-in and
            # explicitly checkpointed for reproducible resume.
            return np.tanh(self._sparse_features(features, weight.shape[0]) @ weight + bias)
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
        public_score_offset = None
        if self.config.get("baseline_offset") == "public_own_score":
            score = observation.get("score") if isinstance(observation, dict) else None
            public_score_offset = score.get("own") if isinstance(score, dict) else None
            if (isinstance(public_score_offset, bool) or
                    not isinstance(public_score_offset, (int, float)) or
                    not math.isfinite(public_score_offset)):
                raise ValueError("public_own_score requires a finite public observation score.own")
        representation = self.config.get("representation", "legacy_json")
        if representation in ("public_v2_split", "public_v3_spatial", "public_v4_contacts"):
            schema = {"public_v2_split": 2, "public_v3_spatial": 3, "public_v4_contacts": 4}[representation]
            if not isinstance(observation, dict) or observation.get("schema_version") != schema or "board" not in observation:
                raise ValueError(f"{representation} requires a schema-v{schema} public observation")
            if any(not isinstance(a, dict) or set(a) != {"action", "public_context"} for a in candidates):
                raise ValueError(f"{representation} requires public candidate views")
            if schema in (3, 4) and any("objective_relations" not in a["public_context"] for a in candidates):
                raise ValueError(f"{representation} requires public objective relations")
            if schema == 4:
                fields = {"intended_endpoint_is_public_objective",
                          "intended_endpoint_is_unowned_public_objective",
                          "acting_unit_is_ground_type1or2"}
                if any(set(a["public_context"].get("objective_contacts", {})) != fields or
                       any(type(v) is not bool for v in a["public_context"]["objective_contacts"].values())
                       for a in candidates):
                    raise ValueError("public_v4_contacts requires Boolean public objective contacts")
            dynamic = {k: v for k, v in observation.items() if k != "board"}
            return Decision(self._split_features(dynamic, observation["board"], cache_second=True),
                            tuple(self._split_features(a["action"], a["public_context"]) for a in candidates), selected, public_score_offset)
        return Decision(self._features(observation), tuple(self._features(a) for a in candidates), selected, public_score_offset)

    def probabilities(self, observation, candidates):
        return self._forward(self._decision(observation, candidates))[3]

    def sample(self, observation, candidates, *, return_probabilities=False):
        record = self._decision(observation, candidates)
        probs = self._forward(record)[3]
        index = int(self.rng.choice(len(candidates), p=probs))
        result = (candidates[index], Decision(record.observation, record.candidates, index, record.public_score_offset))
        return (*result, probs) if return_probabilities else result

    def select(self, observation, candidates):
        return self.sample(observation, candidates)[0]

    __call__ = select

    def _validate_decision_offset(self, decision):
        offset = decision.public_score_offset
        if self.config.get("baseline_offset") == "public_own_score":
            if isinstance(offset, bool) or not isinstance(offset, (int, float)) or not math.isfinite(offset):
                raise ValueError("public_own_score requires an explicitly stored finite decision offset")
        elif offset is not None:
            raise ValueError("public score decision offset requires public_own_score mode")

    def _loss_gradients(self, examples, advantages=None, actor_normalizer=None, baseline_offset=0.0,
                        old_log_probs=None, diagnostics=None):
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
        if self.config.get("baseline_offset") == "public_own_score":
            if baseline_offset != 0.0:
                raise ValueError("public_own_score and prior batch offset are mutually exclusive")
            for decision, _ in examples:
                self._validate_decision_offset(decision)
            # A fixed observation-only baseline, NOT intermediate rewards.
            examples = [(d, score - d.public_score_offset) for d, score in examples]
        # The offset is fixed from prior completed batches, never this batch.
        # These residual targets are internal critic targets, not changed returns.
        if baseline_offset != 0.0:
            examples = [(d, score - baseline_offset) for d, score in examples]
        # Stream activations: retaining every candidate matrix for the whole
        # batch can consume gigabytes on the competition map. Parameters remain
        # unchanged until this method returns, so implicit advantages stay frozen.
        if advantages is not None and len(advantages) != count:
            raise ValueError("advantages must match examples")
        if old_log_probs is not None:
            if len(old_log_probs) != count or not np.isfinite(old_log_probs).all() or (advantages is not None and not np.isfinite(advantages).all()):
                raise ValueError("invalid frozen PPO batch")
        loss = 0.0
        h = self.config["hidden_dim"]
        for i, (decision, score) in enumerate(examples):
            obs, actions, joint, probs, value, log_probs = self._forward(decision)
            adv = score - value if advantages is None else advantages[i]
            actor_weight = adv
            if old_log_probs is None:
                loss += -adv * log_probs[decision.selected] / actor_normalizer + 0.5 * (value - score) ** 2 / count
            else:
                # Compare in log space before exponentiating: favorable extreme
                # ratios are clipped without overflow, tiny probabilities never
                # divide by zero. Unrepresentable adverse objectives fail loudly.
                log_ratio = float(log_probs[decision.selected] - old_log_probs[i])
                clip = self.config["ppo_clip"]
                clipped = ((adv > 0 and log_ratio > math.log1p(clip)) or
                           (adv < 0 and log_ratio < math.log1p(-clip)))
                if clipped:
                    surrogate = adv * (1 + clip if adv > 0 else 1 - clip)
                    actor_weight = 0.0
                elif adv == 0:
                    surrogate = actor_weight = 0.0
                else:
                    try:
                        surrogate = math.copysign(math.exp(math.log(abs(adv)) + log_ratio), adv)
                    except OverflowError as exc:
                        raise FloatingPointError("unrepresentable PPO surrogate") from exc
                    actor_weight = surrogate
                loss += -surrogate / actor_normalizer + 0.5 * (value - score) ** 2 / count
            if diagnostics is not None:
                diagnostics.observe(i, adv, probs, log_probs,
                                    log_ratio if old_log_probs is not None else None,
                                    clipped if old_log_probs is not None else False)
            dlogits = probs.copy()
            dlogits[decision.selected] -= 1
            dlogits *= actor_weight / actor_normalizer
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
            if self.config.get("projection_backend") == "scipy_csr":
                gradients["action"] += self._sparse_features(
                    decision.candidates, self.feature_width).T @ dactions
            else:
                for (indices, values), derivative in zip(decision.candidates, dactions):
                    np.add.at(gradients["action"], indices, values[:, None] * derivative)
        return float(loss), gradients

    def update_batch(self, trajectories, *, diagnostic_seats=None):
        """Update from [(completed own-seat decisions, raw own terminal score)].

        The caller must exclude truncated/unfinished trajectories. No intermediate
        rewards, discount, entropy bonus, return scaling or tactical signal.
        Actor terms sum within each trajectory then average over all completed
        trajectories supplied (including empty ones with zero actor contribution).
        Value regression remains a mean over decisions.
        """
        examples = []
        example_seats = []
        trajectory_count = 0
        scores = []
        for history, score in trajectories:
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("terminal score must be finite")
            if diagnostic_seats is not None:
                if trajectory_count >= len(diagnostic_seats) or diagnostic_seats[trajectory_count] not in ("BLUE", "RED", "UNKNOWN"):
                    raise ValueError("diagnostic seats must match trajectories")
                example_seats.extend([diagnostic_seats[trajectory_count]] * len(history))
            trajectory_count += 1
            scores.append(float(score))
            for decision in history:
                if not isinstance(decision, Decision) or not 0 <= decision.selected < len(decision.candidates):
                    raise ValueError("invalid decision")
                self._validate_decision_offset(decision)
                examples.append((decision, float(score)))
        if diagnostic_seats is not None and len(diagnostic_seats) != trajectory_count:
            raise ValueError("diagnostic seats must match trajectories")
        diagnostics = None
        if diagnostic_seats is not None:
            from .training_diagnostics import EpochDiagnostics
            diagnostics = EpochDiagnostics(example_seats, self.config.get("ppo_clip"))
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
        if self.config.get("update_mode") == "ppo_clip" and examples:
            result = self._update_ppo(examples, trajectory_count, offset, diagnostic_seats=example_seats if diagnostics is not None else None)
            self.return_mean, self.return_count = next_mean, next_count
            return result
        loss, gradients = self._loss_gradients(examples, actor_normalizer=trajectory_count,
                                              baseline_offset=offset, diagnostics=diagnostics)
        if not examples:
            # A completed no-decision episode counts once; an empty batch does
            # nothing. There is no optimizer step when there are no decisions.
            self.return_mean, self.return_count = next_mean, next_count
            return {"decisions": 0, "loss": loss, "gradient_norm": 0.0,
                    **({"diagnostics": {"epochs": [diagnostics.report()]}} if diagnostics is not None else {})}
        norm = self._apply_gradients(loss, gradients)
        self.return_mean, self.return_count = next_mean, next_count
        return {"decisions": len(examples), "loss": loss, "gradient_norm": norm,
                **({"diagnostics": {"epochs": [diagnostics.report()]}} if diagnostics is not None else {})}

    def _apply_gradients(self, loss, gradients):
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
        return norm

    def _update_ppo(self, examples, trajectory_count, offset, *, diagnostic_seats=None):
        # The trainer collects every pending trajectory under unchanged params.
        # Recompute the behavior policy once before ANY optimizer step; never
        # refresh old values/log-probabilities/advantages during these epochs.
        def frozen_prediction(decision):
            forward = self._forward(decision)
            return float(forward[5][decision.selected]), forward[4]
        old = tuple(frozen_prediction(d) for d, _ in examples)
        old_log_probs = tuple(log_prob for log_prob, _ in old)
        old_values = tuple(value for _, value in old)
        advantages = tuple(score - offset - value for (_, score), value in zip(examples, old_values))
        if self.config.get("baseline_offset") == "public_own_score":
            advantages = tuple(score - d.public_score_offset - value
                               for (d, score), value in zip(examples, old_values))
        del old
        backup = ({k: v.copy() for k, v in self.params.items()},
                  {k: v.copy() for k, v in self.m.items()},
                  {k: v.copy() for k, v in self.v.items()}, self.steps)
        epoch_reports = []
        try:
            for _ in range(self.config["update_epochs"]):
                diagnostics = None
                if diagnostic_seats is not None:
                    from .training_diagnostics import EpochDiagnostics
                    diagnostics = EpochDiagnostics(diagnostic_seats, self.config["ppo_clip"])
                loss, gradients = self._loss_gradients(
                    examples, advantages, actor_normalizer=trajectory_count,
                    baseline_offset=offset, old_log_probs=old_log_probs, diagnostics=diagnostics)
                if diagnostics is not None:
                    epoch_reports.append(diagnostics.report())
                norm = self._apply_gradients(loss, gradients)
                if not all(np.isfinite(v).all() for group in (self.params, self.m, self.v) for v in group.values()):
                    raise FloatingPointError("non-finite PPO optimizer state")
        except (FloatingPointError, OverflowError, ValueError):
            self.params, self.m, self.v, self.steps = backup
            raise
        return {"decisions": len(examples), "loss": loss, "gradient_norm": norm,
                "optimizer_steps": self.config["update_epochs"],
                **({"diagnostics": {"epochs": epoch_reports}} if diagnostic_seats is not None else {})}

    @property
    def algorithm(self):
        return "generic_json_ppo_clip" if self.config.get("update_mode") == "ppo_clip" else "generic_json_reinforce"

    def checkpoint(self, *, arrays=False):
        return {"schema_version": 1, "algorithm": self.algorithm,
                "config": dict(self.config), "steps": self.steps,
                "rng": self.rng.bit_generator.state,
                **({"return_baseline": {"mean": self.return_mean, "count": self.return_count}}
                   if self.config.get("baseline_offset") == "prior_batch_mean" else {}),
                **{name: {k: (v if arrays else v.tolist()) for k, v in values.items()}
                   for name, values in (("params", self.params), ("m", self.m), ("v", self.v))}}

    @classmethod
    def from_checkpoint(cls, data):
        if data.get("schema_version") != 1 or data.get("algorithm") not in ("generic_json_reinforce", "generic_json_ppo_clip"):
            raise ValueError("unsupported checkpoint")
        if data["algorithm"] == "generic_json_ppo_clip" and not {"update_mode", "ppo_clip", "update_epochs"} <= data["config"].keys():
            raise ValueError("PPO checkpoint requires explicit update configuration")
        result = cls(**data["config"])
        if data["algorithm"] != result.algorithm:
            raise ValueError("checkpoint algorithm and update mode differ")
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
        if data["steps"] % result.config.get("update_epochs", 1):
            raise ValueError("checkpoint cannot contain a partial PPO epoch batch")
        result.steps = data["steps"]
        result.rng.bit_generator.state = data["rng"]
        return result

    def save(self, path):
        if Path(path).suffix.lower() == ".npz":
            from .compact_checkpoint import save_compact
            return save_compact(path, self.checkpoint(arrays=True))
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(self.checkpoint(), stream, allow_nan=False, separators=(",", ":"))
            stream.write("\n")

    @classmethod
    def load(cls, path):
        from .compact_checkpoint import read_checkpoint
        return cls.from_checkpoint(read_checkpoint(path))
