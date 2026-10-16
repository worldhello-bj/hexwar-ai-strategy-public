"""Dependency-free terminal-score Monte Carlo self-play on synthetic games.

This small baseline memorizes exact observations; it does not generalize, encode
strategy, or claim playing strength. Each seat learns its own raw terminal total
with undiscounted first-visit returns. There are no intermediate reward updates.
Evaluation uses fresh frozen policy copies and seeds excluded from training.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile
import time
import tracemalloc

from engine import scoring
from engine.game import DEPLOY_WINDOWS
from .episode_api import TwoPlayerEpisode
from .match_eval import SIDES, _positive_int, snapshot_manifest, write_synthetic_snapshot

MAX_EPISODES = 10000


def canonical(value):
    """Exact JSON identity; no hidden state or hand-designed domain features."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _seed(*parts):
    return int.from_bytes(hashlib.sha256(canonical(parts).encode()).digest(), "big")


def _epsilon(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("epsilon must be finite and between zero and one")
    return float(value)


class MonteCarloPolicy:
    """Uniform exploration and uniform greedy tie-breaking over legal actions."""

    def __init__(self, *, seed=0, epsilon=0.1, table=None):
        _positive_int("seed", seed, True)
        self.epsilon = _epsilon(epsilon)
        self.rng = random.Random(seed)
        self.table = copy.deepcopy(table) if table is not None else {}

    def __call__(self, observation, legal_actions):
        if not legal_actions:
            raise ValueError("policy needs at least one legal action")
        actions = sorted(legal_actions, key=canonical)
        if self.rng.random() < self.epsilon:
            return copy.deepcopy(self.rng.choice(actions))
        values = self.table.get(canonical(observation), {})
        scores = [values.get(canonical(a), {}).get("value", 0.0) for a in actions]
        best = max(scores)
        return copy.deepcopy(self.rng.choice([a for a, q in zip(actions, scores) if q == best]))

    def update(self, history, terminal_score):
        """Apply one own-seat raw terminal return per distinct state/action pair."""
        if isinstance(terminal_score, bool) or not isinstance(terminal_score, (int, float)) or not math.isfinite(terminal_score):
            raise ValueError("terminal score must be finite")
        for state, action in sorted(set(history)):
            entry = self.table.setdefault(state, {}).setdefault(action, {"count": 0, "value": 0.0})
            entry["count"] += 1
            entry["value"] += (terminal_score - entry["value"]) / entry["count"]


class SelfPlayTrainer:
    """Separate seat tables and histories, reproducible across checkpoint resume."""

    def __init__(self, snapshot, *, seed=0, epsilon=0.1, train_seed_start=0,
                 episode_config=None, allow_real=False, scoring_mode="verified"):
        _positive_int("seed", seed, True)
        _positive_int("train_seed_start", train_seed_start, True)
        if scoring_mode not in ("verified", "platform"):
            raise ValueError("scoring_mode must be 'verified' or 'platform'")
        self.snapshot = str(snapshot)
        self.allow_real = bool(allow_real)
        self.manifest = snapshot_manifest(snapshot, allow_real=self.allow_real)
        self.seed = seed
        self.scoring_mode = scoring_mode
        self.epsilon = _epsilon(epsilon)
        self.train_seed_start = train_seed_start
        self.episode_config = copy.deepcopy(episode_config or {})
        # 奖励口径随 episode_config 一起检查点化（决定哪些分项能进终局奖励）。
        self.episode_config["scoring_mode"] = scoring_mode
        # Validate configuration even for an empty run.
        TwoPlayerEpisode(self.snapshot, **self.episode_config)
        self.policies = {s: MonteCarloPolicy(epsilon=self.epsilon) for s in SIDES}
        self.episodes_completed = 0

    def _run(self, episode_seed, *, training):
        if snapshot_manifest(self.snapshot, allow_real=self.allow_real) != self.manifest:
            raise ValueError("snapshot changed")
        episode = TwoPlayerEpisode(self.snapshot, **self.episode_config)
        status = episode.reset(seed=episode_seed)
        # Fresh RNG and detached tables for every seat and episode. Evaluation
        # cannot change training tables, history, counters or random streams.
        policies = {s: MonteCarloPolicy(seed=_seed(self.seed, episode_seed, s, training),
                                      epsilon=self.epsilon if training else 0,
                                      table=self.policies[s].table) for s in SIDES}
        histories = {s: [] for s in SIDES}
        while not status["terminated"] and not status["truncated"]:
            side = status["current_actor"]
            observation = episode.observe(side)
            legal = episode.legal_actions(side)
            if not legal:
                status = {**status, "terminated": False, "truncated": True,
                          "reason": "no_legal_actions", "final_scores": None,
                          "rewards": {s: 0 for s in SIDES}}
                break
            action = policies[side](observation, legal)
            histories[side].append((canonical(observation), canonical(action)))
            status = episode.step(action, side=side)
        if training and status["terminated"] and not status["truncated"]:
            scores = status["final_scores"]
            # Validate both returns before mutating either table.
            if not isinstance(scores, dict) or any(isinstance(scores.get(s), bool)
                    or not isinstance(scores.get(s), (int, float))
                    or not math.isfinite(scores[s]) for s in SIDES):
                raise ValueError("completed episode requires both finite final scores")
            for side in SIDES:
                self.policies[side].update(histories[side], scores[side])
        return {"seed": episode_seed, "training": training, "status": status,
                "decisions": {s: len(histories[s]) for s in SIDES}}

    def train(self, episodes=1):
        _positive_int("episodes", episodes, True)
        if episodes > MAX_EPISODES:
            raise ValueError("episodes exceeds per-call safety bound")
        records = []
        for _ in range(episodes):
            records.append(self._run(self.train_seed_start + self.episodes_completed, training=True))
            self.episodes_completed += 1
        return records

    def evaluate(self, seeds):
        seeds = list(seeds)
        if not seeds or len(seeds) > MAX_EPISODES or len(set(seeds)) != len(seeds):
            raise ValueError("evaluation needs bounded, nonempty unique seeds")
        for seed in seeds:
            _positive_int("evaluation seed", seed, True)
            if self.train_seed_start <= seed < self.train_seed_start + self.episodes_completed:
                raise ValueError("evaluation seed overlaps training")
        return [self._run(seed, training=False) for seed in seeds]

    def checkpoint(self):
        return copy.deepcopy({"schema_version": 1, "algorithm": "first_visit_terminal_mc",
                              "seed": self.seed, "epsilon": self.epsilon,
                              "train_seed_start": self.train_seed_start,
                              "episodes_completed": self.episodes_completed,
                              # 计分口径绑定：口径变了，表里的 return 就不可比。
                              "scoring_version": scoring.SCORING_VERSION,
                              "scoring_mode": self.scoring_mode,
                              "snapshot": self.manifest, "episode_config": self.episode_config,
                              "tables": {s: self.policies[s].table for s in SIDES}})

    def save(self, path):
        """Write a new checkpoint; never silently overwrite an existing file."""
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(self.checkpoint(), stream, sort_keys=True, allow_nan=False)
            stream.write("\n")

    @classmethod
    def load(cls, path, snapshot, *, allow_real=False):
        data = json.loads(Path(path).read_text("utf-8"))
        if data.get("schema_version") != 1 or data.get("algorithm") != "first_visit_terminal_mc":
            raise ValueError("unsupported checkpoint")
        # 旧检查点（没有 scoring_version）是在旧计分口径下学的：直接拒绝，不混用。
        if data.get("scoring_version") != scoring.SCORING_VERSION:
            raise ValueError(
                "checkpoint scoring version %r does not match %r; retrain or convert"
                % (data.get("scoring_version"), scoring.SCORING_VERSION))
        trainer = cls(snapshot, seed=data["seed"], epsilon=data["epsilon"],
                      train_seed_start=data["train_seed_start"],
                      episode_config=data["episode_config"], allow_real=allow_real,
                      scoring_mode=data.get("scoring_mode", "verified"))
        if trainer.manifest != data["snapshot"]:
            raise ValueError("checkpoint snapshot does not match")
        _positive_int("episodes_completed", data["episodes_completed"], True)
        trainer.episodes_completed = data["episodes_completed"]
        if set(data["tables"]) != set(SIDES):
            raise ValueError("checkpoint must contain both seat tables")
        for side in SIDES:
            table = data["tables"][side]
            for state, actions in table.items():
                json.loads(state)
                for action, entry in actions.items():
                    json.loads(action)
                    _positive_int("visit count", entry["count"])
                    if isinstance(entry["value"], bool) or not isinstance(entry["value"], (float, int)) or not math.isfinite(entry["value"]):
                        raise ValueError("invalid checkpoint action value")
            trainer.policies[side].table = copy.deepcopy(table)
        return trainer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-memory", action="store_true", help="opt-in Python allocation tracing (slower)")
    parser.add_argument("--snapshot", type=Path, help="original synthetic snapshot with provenance; default: tiny generated fixture")
    parser.add_argument("--allow-real-map", action="store_true",
                        help="显式允许使用真实想定（scenario/7527 随仓库公开；本地完整采集 snapshot/parsed 不入库）")
    parser.add_argument("--move-radius", type=int, default=None,
                        help="覆盖机动枚举半径（默认 1；真实地图建议 3~6）")
    parser.add_argument("--deep-validate", action="store_true",
                        help="用参考口径逐候选事务校验枚举（慢，仅供差分复现）")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--scoring", dest="scoring_mode", choices=("verified", "platform"),
                        default="verified",
                        help="奖励口径：verified（默认，只用已实测分项）/ platform（平台式三项和）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[1000000, 1000001])
    parser.add_argument("--max-seconds", type=int, default=60)
    parser.add_argument("--deploy-window", choices=DEPLOY_WINDOWS, default="opening",
                        help="巡飞弹部署（上车）窗口：opening=对抗开始前（手册 4.10(3)，默认）；"
                             "any=未发射前随时（训练里要学到用弹就选它）")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--checkpoint", type=Path, required=True, help="new output checkpoint")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--report", type=Path, help="optional new JSON report")
    args = parser.parse_args(argv)
    if args.checkpoint.exists() or (args.report and args.report.exists()):
        parser.error("output exists; choose new filenames")
    if args.report and args.report.resolve() == args.checkpoint.resolve():
        parser.error("report and checkpoint must be different files")
    try:
        with tempfile.TemporaryDirectory(prefix="synthetic-selfplay-") as tmp:
            snapshot = args.snapshot or write_synthetic_snapshot(Path(tmp) / "fixture")
            episode_config = {"max_seconds": args.max_seconds,
                              "deploy_window": args.deploy_window,
                              "max_steps": args.max_steps}
            if args.move_radius is not None:
                episode_config["move_radius"] = args.move_radius
            if args.deep_validate:
                episode_config["deep_validate"] = True
            trainer = (SelfPlayTrainer.load(args.resume, snapshot, allow_real=args.allow_real_map)
                       if args.resume else
                       SelfPlayTrainer(snapshot, seed=args.seed, epsilon=args.epsilon,
                                       train_seed_start=args.train_seed_start,
                                       episode_config=episode_config,
                                       allow_real=args.allow_real_map,
                                       scoring_mode=args.scoring_mode))
            if args.scoring_mode != trainer.scoring_mode:
                raise ValueError("requested scoring mode differs from resumed checkpoint")
            # Reject heldout overlap with the whole planned run before training.
            if any(trainer.train_seed_start <= s < trainer.train_seed_start + trainer.episodes_completed + args.episodes for s in args.eval_seeds):
                raise ValueError("evaluation seeds overlap planned training")
            start = time.perf_counter()
            if args.profile_memory:
                tracemalloc.start()
            try:
                records = trainer.train(args.episodes)
                evaluation = trainer.evaluate(args.eval_seeds)
                peak_bytes = tracemalloc.get_traced_memory()[1] if args.profile_memory else None
            finally:
                if args.profile_memory:
                    tracemalloc.stop()
            resources = {"elapsed_seconds": time.perf_counter() - start,
                         "peak_traced_python_bytes": peak_bytes}
            try:
                import resource
                import sys
                # ru_maxrss is bytes on macOS and KiB on Linux; process lifetime peak.
                resources["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
            except ImportError:
                resources["process_peak_rss_bytes"] = None

            trainer.save(args.checkpoint)
            report = {"algorithm": "first_visit_terminal_mc", "training": records,
                      "evaluation": evaluation, "episodes_completed": trainer.episodes_completed,
                      "scoring_mode": trainer.scoring_mode,
                      "scoring_version": scoring.SCORING_VERSION,
                      "resources": resources,
                      "caveat": "Exact-observation baseline; evaluation is trained-vs-trained self-play, not a comparison against a random or strong opponent and establishes no playing-strength claim."}
            if args.report:
                with args.report.open("x", encoding="utf-8") as stream:
                    json.dump(report, stream, indent=2, allow_nan=False)
                    stream.write("\n")
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"episodes_completed": trainer.episodes_completed,
                      "completed_training": sum(r["status"]["terminated"] for r in records),
                      "truncated_training": sum(r["status"]["truncated"] for r in records),
                      "evaluation_episodes": len(evaluation), "resources": resources}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
