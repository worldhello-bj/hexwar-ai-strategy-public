"""Held-out frozen Monte Carlo versus uniform-random synthetic game evaluation.

Each seed plays both policy seats. Results use raw terminal scores, never a
step-budget truncation. This descriptive comparison makes no strength claim.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from .episode_api import TwoPlayerEpisode
from .match_eval import SIDES, _positive_int, random_policy, snapshot_manifest
from .selfplay import MonteCarloPolicy, SelfPlayTrainer, _seed, canonical

MAX_SEEDS = 256
MAX_STEPS = 10000


def _summary(records):
    counts = {label: sum(r["result"] == label for r in records)
              for label in ("win", "draw", "loss")}
    completed = sum(counts.values())
    return {"games": len(records), "completed": completed,
            "truncated": len(records) - completed, **counts,
            "rates": {label: count / completed if completed else None
                      for label, count in counts.items()}}


def evaluate_checkpoint(checkpoint, snapshot, seeds, *, allow_real=False):
    """Evaluate fresh independent policies; retain checkpoint rules and seat tables.

    The checkpoint's training interval is excluded. A policy playing RED uses
    the trained RED table, not BLUE's observation/action keys relabeled as RED.
    Neither checkpoint bytes nor loaded training tables are updated.
    """
    seeds = list(seeds)
    if not seeds or len(seeds) > MAX_SEEDS:
        raise ValueError("evaluation requires 1 to 256 seeds")
    for seed in seeds:
        _positive_int("evaluation seed", seed, True)
    if len(set(seeds)) != len(seeds):
        raise ValueError("evaluation seeds must be unique")
    checkpoint = Path(checkpoint)
    checkpoint_bytes = checkpoint.read_bytes()
    trainer = SelfPlayTrainer.load(checkpoint, snapshot, allow_real=allow_real)
    before = canonical(trainer.checkpoint())
    config = trainer.episode_config
    if not config.get("public_score", True):
        raise ValueError("evaluation requires the public totals scoreboard")
    if config.get("max_steps", 100) > MAX_STEPS:
        raise ValueError("checkpoint exceeds evaluation step safety bound")
    start = trainer.train_seed_start
    end = start + trainer.episodes_completed
    if any(start <= seed < end for seed in seeds):
        raise ValueError("evaluation seed overlaps training")
    records = []
    for seed in seeds:
        for policy_side in SIDES:
            if snapshot_manifest(snapshot, allow_real=allow_real) != trainer.manifest:
                raise ValueError("snapshot changed")
            opponent_side = next(s for s in SIDES if s != policy_side)
            policies = {
                policy_side: MonteCarloPolicy(
                    seed=_seed(trainer.seed, seed, policy_side, "frozen"),
                    epsilon=0, table=trainer.policies[policy_side].table),
                opponent_side: random_policy(
                    _seed(trainer.seed, seed, policy_side, "random")),
            }
            episode = TwoPlayerEpisode(snapshot, **config)
            status = episode.reset(seed=seed)
            while not status["terminated"] and not status["truncated"]:
                side = status["current_actor"]
                legal = episode.legal_actions(side)
                if not legal:
                    status = {**status, "terminated": False, "truncated": True,
                              "reason": "no_legal_actions", "winner": None,
                              "final_scores": None, "rewards": {s: 0 for s in SIDES}}
                    break
                status = episode.step(policies[side](episode.observe(side), legal), side=side)
            result = None
            if status["terminated"] and not status["truncated"]:
                scores = status["final_scores"]
                if not isinstance(scores, dict) or any(
                        isinstance(scores.get(s), bool)
                        or not isinstance(scores.get(s), (int, float))
                        or not math.isfinite(scores[s]) for s in SIDES):
                    raise ValueError("completed game requires finite terminal scores")
                diff = scores[policy_side] - scores[opponent_side]
                result = "win" if diff > 0 else "loss" if diff < 0 else "draw"
            records.append({"seed": seed, "policy_side": policy_side,
                            "random_side": opponent_side, "result": result,
                            "status": status})
    if canonical(trainer.checkpoint()) != before or checkpoint.read_bytes() != checkpoint_bytes:
        raise RuntimeError("checkpoint changed during evaluation")
    if snapshot_manifest(snapshot, allow_real=allow_real) != trainer.manifest:
        raise ValueError("snapshot changed")
    return {"schema_version": 1, "evaluation": "frozen_mc_vs_uniform_random",
            "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            "snapshot": trainer.manifest, "episode_config": config,
            "training_seed_interval": {"start_inclusive": start, "end_exclusive": end},
            "heldout_seeds": seeds, "summary": _summary(records),
            "by_policy_seat": {s: _summary([r for r in records if r["policy_side"] == s])
                               for s in SIDES}, "games": records,
            "caveat": "Descriptive synthetic-game sample only; no playing-strength claim. "
                      "Rates exclude truncations. Paired seats share a scenario seed and "
                      "are not independent samples. Unseen observations use uniform greedy ties."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True, help="new JSON output; never overwritten")
    parser.add_argument("--allow-real-map", action="store_true",
                        help="显式允许使用真实想定（scenario/7527 随仓库公开；本地完整采集 snapshot/parsed 不入库）")
    args = parser.parse_args(argv)
    if args.report.exists():
        parser.error("report exists; choose a new filename")
    try:
        report = evaluate_checkpoint(args.checkpoint, args.snapshot, args.seeds,
                                     allow_real=args.allow_real_map)
        with args.report.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
