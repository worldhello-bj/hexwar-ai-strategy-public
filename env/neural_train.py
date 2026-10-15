"""Generic terminal-return neural learning in an offline synthetic game.

One shared learner alternates seats against a frozen uniform legal-action
sampler. Only own-seat observations enter the learner. Rewards are the original
terminal own total, undiscounted, with no shaping or coordinate transformations.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np

from engine import scoring
from .episode_api import TwoPlayerEpisode
from .match_eval import SIDES, _positive_int, snapshot_manifest
from .neural_policy import NeuralPolicy, Decision
from .public_observation import PublicObservationAdapter


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def derived_seed(*parts):
    return int.from_bytes(hashlib.sha256(canonical(parts).encode()).digest()[:8], "big")


def summary(records):
    completed = [r for r in records if r["own_score"] is not None]
    return {"games": len(records), "completed": len(completed),
            "truncated": len(records) - len(completed),
            "mean_own_score": float(np.mean([r["own_score"] for r in completed])) if completed else None,
            "mean_score_diff": float(np.mean([r["score_diff"] for r in completed])) if completed else None,
            **{label: sum(r["result"] == label for r in completed) for label in ("win", "draw", "loss")}}


class NeuralTrainer:
    """Resumable on-policy batches; incomplete batches survive checkpoints.

    A train call never flushes a partial batch: splitting the same episode budget
    across calls does not alter optimization. Evaluation seeds are reserved at
    construction and cannot subsequently be used as training input.
    """
    def __init__(self, snapshot, *, seed=0, train_seed_start=0, batch_episodes=4,
                 episode_config=None, policy_config=None, heldout_seeds=(1000000, 1000001),
                 observation_schema=1, allow_real=False, scoring_mode="verified"):
        for name, value in (("seed", seed), ("train_seed_start", train_seed_start)):
            _positive_int(name, value, True)
        _positive_int("batch_episodes", batch_episodes)
        if scoring_mode not in ("verified", "platform"):
            raise ValueError("scoring_mode must be 'verified' or 'platform'")
        self.snapshot = str(snapshot)
        self.allow_real = bool(allow_real)
        self.manifest = snapshot_manifest(snapshot, allow_real=self.allow_real)
        self.seed, self.train_seed_start, self.batch_episodes = seed, train_seed_start, batch_episodes
        self.episode_config = {"max_seconds": 1800, "max_steps": 1000, "public_score": True}
        self.episode_config.update(copy.deepcopy(episode_config or {}))
        # 奖励口径随 episode_config 一起检查点化：它决定了哪些分项能进"终局原始分"。
        self.episode_config["scoring_mode"] = scoring_mode
        self.scoring_mode = scoring_mode
        TwoPlayerEpisode(snapshot, **self.episode_config)
        self.heldout_seeds = list(heldout_seeds)
        if not self.heldout_seeds or len(self.heldout_seeds) > 256 or len(set(self.heldout_seeds)) != len(self.heldout_seeds):
            raise ValueError("reserve 1 to 256 unique heldout seeds")
        for value in self.heldout_seeds:
            _positive_int("heldout seed", value, True)
        if isinstance(observation_schema, bool) or not isinstance(observation_schema, int) or observation_schema not in (1, 2):
            raise ValueError("observation_schema must be 1 or 2")
        self.observation_schema = observation_schema
        self.policy_config = copy.deepcopy(policy_config or {})
        representation = "public_v2_split" if observation_schema == 2 else "legacy_json"
        if self.policy_config.get("representation", representation) != representation:
            raise ValueError("policy representation does not match observation schema")
        if observation_schema == 2:
            self.policy_config["representation"] = representation
        self.policy = NeuralPolicy(seed=seed, **self.policy_config)
        self.initial_policy = copy.deepcopy(self.policy.checkpoint())
        self.episodes_attempted = 0
        self.episodes_completed = 0
        self.pending = []
        self.updates = 0

    def _rollout(self, policy, episode_seed, learner_side, *, training):
        if snapshot_manifest(self.snapshot, allow_real=self.allow_real) != self.manifest:
            raise ValueError("snapshot changed")
        start = time.perf_counter()
        episode = TwoPlayerEpisode(self.snapshot, **self.episode_config)
        public_episode = PublicObservationAdapter(episode) if self.observation_schema == 2 else episode
        status = episode.reset(seed=episode_seed)
        opponent_rng = random.Random(derived_seed(self.seed, episode_seed, learner_side, "opponent"))
        eval_rng = np.random.default_rng(derived_seed(self.seed, episode_seed, learner_side, "evaluation"))
        trajectory, entropies, counts = [], [], []
        decisions = 0
        while not status["terminated"] and not status["truncated"]:
            side = status["current_actor"]
            candidates = (public_episode.candidate_views(side)
                          if side == learner_side and self.observation_schema == 2
                          else episode.legal_actions(side))
            if not candidates:
                status = {**status, "terminated": False, "truncated": True,
                          "reason": "no_legal_actions", "winner": None, "final_scores": None,
                          "rewards": {s: 0 for s in SIDES}}
                break
            if side == learner_side:
                observation = public_episode.observe(side)
                probs = policy.probabilities(observation, candidates)
                entropies.append(float(-np.sum(probs * np.log(np.maximum(probs, 1e-300)))))
                counts.append(len(candidates))
                decisions += 1
                if training:
                    action, decision = policy.sample(observation, candidates)
                    trajectory.append(decision)
                else:
                    action = candidates[int(eval_rng.choice(len(candidates), p=probs))]
            else:
                # No opposing observation is ever fetched or supplied to learner.
                action = opponent_rng.choice(candidates)
            if side == learner_side and self.observation_schema == 2:
                action = action["action"]
            status = episode.step(action, side=side)
            if not status["terminated"] and any(status["rewards"].values()):
                raise ValueError("episode violated terminal-only reward contract")
        own_score = diff = result = None
        if status["terminated"] and not status["truncated"]:
            scores = status["final_scores"]
            if not isinstance(scores, dict) or any(isinstance(scores.get(s), bool) or
                    not isinstance(scores.get(s), (int, float)) or not math.isfinite(scores[s]) for s in SIDES):
                raise ValueError("completed episode requires finite terminal scores")
            own_score = scores[learner_side]
            other = SIDES[1 - SIDES.index(learner_side)]
            diff = own_score - scores[other]
            result = "win" if diff > 0 else "loss" if diff < 0 else "draw"
        record = {"seed": episode_seed, "learner_side": learner_side, "status": status,
                  "own_score": own_score, "score_diff": diff, "result": result,
                  "decisions": decisions, "mean_entropy": float(np.mean(entropies)) if entropies else None,
                  "mean_candidates": float(np.mean(counts)) if counts else None,
                  "max_candidates": max(counts, default=0), "elapsed_seconds": time.perf_counter() - start}
        return record, trajectory

    def train(self, episodes=20):
        _positive_int("episodes", episodes, True)
        if episodes > 10000:
            raise ValueError("episodes exceeds per-call safety bound")
        first = self.train_seed_start + self.episodes_attempted
        if any(first <= s < first + episodes for s in self.heldout_seeds):
            raise ValueError("training overlaps reserved heldout seeds")
        records = []
        for _ in range(episodes):
            index = self.episodes_attempted
            # PPO updates are atomic. Preserve the final rollout's sampling RNG
            # too, so a failed batch can be retried or checkpointed exactly.
            ppo_rng = (copy.deepcopy(self.policy.rng.bit_generator.state)
                       if self.policy.config.get("update_mode") == "ppo_clip" else None)
            record, trajectory = self._rollout(self.policy, self.train_seed_start + index,
                                               SIDES[index % 2], training=True)
            self.episodes_attempted += 1
            if record["own_score"] is not None:
                self.episodes_completed += 1
                self.pending.append((trajectory, record["own_score"]))
                if len(self.pending) == self.batch_episodes:
                    try:
                        record["update"] = self.policy.update_batch(self.pending)
                    except (FloatingPointError, OverflowError, ValueError):
                        if ppo_rng is not None:
                            self.pending.pop()
                            self.episodes_attempted -= 1
                            self.episodes_completed -= 1
                            self.policy.rng.bit_generator.state = ppo_rng
                        raise
                    self.pending.clear()
                    self.updates += 1
            records.append(record)
        return records

    def evaluate(self, seeds=None):
        seeds = self.heldout_seeds if seeds is None else list(seeds)
        if not seeds or len(set(seeds)) != len(seeds) or any(s not in self.heldout_seeds for s in seeds):
            raise ValueError("evaluation must use unique reserved heldout seeds")
        before = canonical(self.checkpoint())
        reports = {}
        for label, checkpoint in (("initial", self.initial_policy), ("trained", self.policy.checkpoint())):
            policy = NeuralPolicy.from_checkpoint(copy.deepcopy(checkpoint))
            frozen = canonical(policy.checkpoint())
            games = [self._rollout(policy, seed, side, training=False)[0] for seed in seeds for side in SIDES]
            if canonical(policy.checkpoint()) != frozen:
                raise RuntimeError("evaluation mutated frozen policy")
            reports[label] = {"summary": summary(games), "by_seat": {
                side: summary([r for r in games if r["learner_side"] == side]) for side in SIDES}, "games": games}
        if canonical(self.checkpoint()) != before:
            raise RuntimeError("evaluation mutated trainer")
        reports["heldout_seeds"] = list(seeds)
        reports["comparison"] = "Frozen initial and trained stochastic policies versus uniform random; identical paired seeds and seats."
        reports["caveat"] = "Descriptive synthetic-game sample, not a strength claim; truncations excluded from score and W/D/L summaries."
        return reports

    @property
    def algorithm(self):
        return "neural_terminal_ppo_clip_vs_random" if self.policy.config.get("update_mode") == "ppo_clip" else "neural_terminal_reinforce_vs_random"

    def checkpoint(self):
        return copy.deepcopy({"schema_version": 1, "algorithm": self.algorithm,
            **({"observation_schema": 2} if self.observation_schema == 2 else {}),
            # 计分口径绑定：口径一变，旧检查点的"原始终局分"就不可比，
            # 混用两套奖励会污染训练，所以载入时必须逐步校验。
            "scoring_version": scoring.SCORING_VERSION,
            "scoring_mode": self.scoring_mode,
            "snapshot": self.manifest, "seed": self.seed, "train_seed_start": self.train_seed_start,
            "batch_episodes": self.batch_episodes, "episode_config": self.episode_config,
            "policy_config": self.policy_config, "heldout_seeds": self.heldout_seeds,
            "episodes_attempted": self.episodes_attempted, "episodes_completed": self.episodes_completed,
            "updates": self.updates, "policy": self.policy.checkpoint(), "initial_policy": self.initial_policy,
            "pending": [{"decisions": [d.to_dict() for d in trajectory], "return": score}
                        for trajectory, score in self.pending]})

    def save(self, path):
        with Path(path).open("x", encoding="utf-8") as stream:
            stream.write(canonical(self.checkpoint()) + "\n")

    @classmethod
    def load(cls, path, snapshot, *, allow_real=False):
        data = json.loads(Path(path).read_text("utf-8"))
        if data.get("schema_version") != 1 or data.get("algorithm") not in ("neural_terminal_reinforce_vs_random", "neural_terminal_ppo_clip_vs_random"):
            raise ValueError("unsupported checkpoint")
        # 旧检查点（没有 scoring_version）是在旧计分口径下训练的，奖励不可比：
        # 直接拒绝，而不是静默拿它继续跑。
        if data.get("scoring_version") != scoring.SCORING_VERSION:
            raise ValueError(
                "checkpoint scoring version %r does not match %r; retrain or convert"
                % (data.get("scoring_version"), scoring.SCORING_VERSION))
        trainer = cls(snapshot, observation_schema=data.get("observation_schema", 1), allow_real=allow_real,
                      scoring_mode=data.get("scoring_mode", "verified"),
                      **{k: data[k] for k in ("seed", "train_seed_start", "batch_episodes",
                                              "episode_config", "policy_config", "heldout_seeds")})
        if trainer.scoring_mode != data.get("scoring_mode", "verified"):
            raise ValueError("checkpoint scoring mode differs")
        if data["algorithm"] != trainer.algorithm:
            raise ValueError("checkpoint algorithm and update mode differ")
        if trainer.manifest != data["snapshot"]:
            raise ValueError("checkpoint snapshot does not match")
        for key in ("episodes_attempted", "episodes_completed", "updates"):
            _positive_int(key, data[key], True)
            setattr(trainer, key, data[key])
        if trainer.episodes_completed > trainer.episodes_attempted:
            raise ValueError("invalid episode counters")
        trainer.policy = NeuralPolicy.from_checkpoint(data["policy"])
        initial = NeuralPolicy.from_checkpoint(data["initial_policy"])
        expected_config = NeuralPolicy(seed=trainer.seed, **trainer.policy_config).config
        if trainer.policy.config != expected_config or initial.config != expected_config or initial.steps != 0:
            raise ValueError("checkpoint policy configuration or initialization differs")
        epochs = trainer.policy.config.get("update_epochs", 1)
        if trainer.policy.steps > trainer.updates * epochs or trainer.policy.steps % epochs:
            raise ValueError("optimizer steps exceed completed batch updates")
        if trainer.policy.config.get("baseline_offset") == "prior_batch_mean":
            if (initial.return_count != 0 or
                    trainer.policy.return_count != trainer.updates * trainer.batch_episodes):
                raise ValueError("return baseline count differs from completed batches")
        trainer.initial_policy = copy.deepcopy(data["initial_policy"])
        trainer.pending = [([Decision.from_dict(d) for d in row["decisions"]], row["return"]) for row in data["pending"]]
        if len(trainer.pending) >= trainer.batch_episodes or trainer.episodes_completed != trainer.updates * trainer.batch_episodes + len(trainer.pending):
            raise ValueError("invalid batch counters")
        buckets = trainer.policy.feature_width
        if any((features[0] >= buckets).any() for history, _ in trainer.pending for decision in history
               for features in (decision.observation, *decision.candidates)):
            raise ValueError("pending feature index exceeds policy buckets")
        if any(isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
               for _, score in trainer.pending):
            raise ValueError("invalid pending return")
        if any(trainer.train_seed_start <= s < trainer.train_seed_start + trainer.episodes_attempted for s in trainer.heldout_seeds):
            raise ValueError("checkpoint training overlaps heldout seeds")
        return trainer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path,
                        help="想定目录（比赛初设：scenario/7527；本地完整采集：snapshot/parsed）")
    parser.add_argument("--allow-real-map", action="store_true",
                        help="显式允许使用真实想定（scenario/7527 随仓库公开；本地完整采集 snapshot/parsed 不入库）")
    parser.add_argument("--move-radius", type=int, default=None,
                        help="机动枚举半径（默认 1；真实地图建议 3~6）")
    parser.add_argument("--deep-validate", action="store_true",
                        help="用参考口径逐候选事务校验枚举（慢，仅供差分复现）")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[1000000, 1000001])
    parser.add_argument("--batch-episodes", type=int, default=4)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--buckets", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--observation-schema", type=int, choices=(1, 2), default=None,
                        help="2 opts into public map/dynamic/candidate split; resume inherits saved schema")
    parser.add_argument("--baseline-offset", choices=("none", "prior_batch_mean"), default=None,
                        help="opt-in prior completed-batch score mean; resume inherits saved mode")
    parser.add_argument("--update-mode", choices=("reinforce", "ppo_clip"), default=None)
    parser.add_argument("--scoring", dest="scoring_mode", choices=("verified", "platform"),
                        default="verified",
                        help="reward scope: verified = only confirmed components (default); "
                             "platform = use the platform-style three-component sum")
    parser.add_argument("--ppo-clip", type=float, default=None)
    parser.add_argument("--update-epochs", type=int, default=None)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.checkpoint.exists() or args.report.exists() or args.checkpoint.resolve() == args.report.resolve():
        parser.error("choose separate new checkpoint and report paths")
    try:
        episode_config = {"max_seconds": args.max_seconds, "max_steps": args.max_steps}
        if args.move_radius is not None:
            episode_config["move_radius"] = args.move_radius
        if args.deep_validate:
            episode_config["deep_validate"] = True
        trainer = NeuralTrainer.load(args.resume, args.snapshot,
                                    allow_real=args.allow_real_map) if args.resume else NeuralTrainer(
            args.snapshot, observation_schema=args.observation_schema or 1, seed=args.seed, train_seed_start=args.train_seed_start, batch_episodes=args.batch_episodes,
            heldout_seeds=args.eval_seeds, episode_config=episode_config,
            allow_real=args.allow_real_map, scoring_mode=args.scoring_mode,
            policy_config={"buckets": args.buckets, "hidden_dim": args.hidden_dim, "learning_rate": args.learning_rate,
                           **({"baseline_offset": args.baseline_offset} if args.baseline_offset is not None else {}),
                           **{k: getattr(args, k) for k in ("update_mode", "ppo_clip", "update_epochs") if getattr(args, k) is not None}})
        if args.observation_schema is not None and args.observation_schema != trainer.observation_schema:
            raise ValueError("requested observation schema differs from resumed checkpoint")
        if args.scoring_mode != trainer.scoring_mode:
            raise ValueError("requested scoring mode differs from resumed checkpoint")
        if (args.baseline_offset is not None and
                args.baseline_offset != trainer.policy.config.get("baseline_offset", "none")):
            raise ValueError("requested baseline offset differs from resumed checkpoint")
        for key in ("update_mode", "ppo_clip", "update_epochs"):
            requested = getattr(args, key)
            if requested is not None and requested != trainer.policy.config.get(key, "reinforce" if key == "update_mode" else None):
                raise ValueError("requested " + key + " differs from resumed checkpoint")
        start = time.perf_counter()
        records = trainer.train(args.episodes)
        evaluation = trainer.evaluate()
        trainer.save(args.checkpoint)
        resources = {"elapsed_seconds": time.perf_counter() - start}
        try:
            import resource
            import sys
            resources["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        except ImportError:
            resources["process_peak_rss_bytes"] = None
        report = {"algorithm": trainer.algorithm, "snapshot": trainer.manifest,
                  "observation_schema": trainer.observation_schema,
                  "scoring_mode": trainer.scoring_mode,
                  "scoring_version": scoring.SCORING_VERSION,
                  "episode_config": trainer.episode_config, "policy_config": trainer.policy_config,
                  "training": records, "training_summary": summary(records), "evaluation": evaluation,
                  "episodes_attempted": trainer.episodes_attempted, "episodes_completed": trainer.episodes_completed,
                  "updates": trainer.updates, "pending_completed_episodes": len(trainer.pending), "resources": resources}
        with args.report.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"training": report["training_summary"], "evaluation": {k: evaluation[k]["summary"] for k in ("initial", "trained")}, "resources": resources}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
