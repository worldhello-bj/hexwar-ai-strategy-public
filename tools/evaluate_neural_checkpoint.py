#!/usr/bin/env python3
"""Bounded fresh-seed evaluation of frozen initial/trained synthetic-game models.

No training or checkpoint changes. Each seed runs both policy seats against
uniform random, with identical scenario and sampling seeds for both models.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.match_eval import SIDES, _positive_int
from env.neural_train import NeuralTrainer, canonical


def paired_deltas(evaluation, seeds):
    """Use scenario seeds as sampling units, averaging the two correlated seats.

    A seed is excluded from the paired statistic if either model/seat truncates.
    No outcome bound is assumed and no normal confidence interval is claimed.
    """
    indexed = {label: {(r["seed"], r["learner_side"]): r for r in evaluation[label]["games"]}
               for label in ("initial", "trained")}
    pairs = []
    for seed in seeds:
        rows = {label: [indexed[label][seed, side] for side in SIDES] for label in indexed}
        complete = all(r["own_score"] is not None for group in rows.values() for r in group)
        seat_deltas = {side: indexed["trained"][seed, side]["own_score"] - indexed["initial"][seed, side]["own_score"]
                       for side in SIDES} if complete else None
        pairs.append({"seed": seed, "complete": complete,
                      "own_score_delta_by_seat": seat_deltas,
                      "mean_own_score_delta": statistics.mean(seat_deltas.values()) if complete else None})
    values = [r["mean_own_score_delta"] for r in pairs if r["complete"]]
    return {"unit": "scenario seed; mean of BLUE and RED own-score deltas",
            "delta_direction": "trained minus initial",
            "seed_pairs": len(seeds), "complete_seed_pairs": len(values),
            "excluded_seed_pairs": len(seeds) - len(values),
            "mean_own_score_delta": statistics.mean(values) if values else None,
            "sample_standard_error": statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None,
            "pairs": pairs,
            "caveat": "Descriptive sample standard error across scenario seeds, not a confidence interval or proof of improvement. Both seats of a seed are correlated; incomplete pairs are excluded."}


def evaluate_checkpoint(checkpoint, snapshot, *, seed_start, seed_count):
    _positive_int("seed_start", seed_start, True)
    _positive_int("seed_count", seed_count)
    if seed_count > 256:
        raise ValueError("seed_count exceeds 256-seed evaluation safety bound")
    checkpoint = Path(checkpoint)
    original_bytes = checkpoint.read_bytes()
    trainer = NeuralTrainer.load(checkpoint, snapshot)
    seeds = list(range(seed_start, seed_start + seed_count))
    start = trainer.train_seed_start
    end = start + trainer.episodes_attempted
    if any(start <= seed < end for seed in seeds):
        raise ValueError("fresh evaluation seeds overlap the full training attempt interval")
    if set(seeds).intersection(trainer.heldout_seeds):
        raise ValueError("fresh evaluation seeds overlap checkpoint's previously reserved heldout seeds")
    # This object is an independent in-memory clone loaded from the checkpoint.
    # Change its evaluation reservation only after validating the whole interval.
    original_reservation = list(trainer.heldout_seeds)
    trainer.heldout_seeds = seeds
    frozen = canonical(trainer.checkpoint())
    started = time.perf_counter()
    evaluation = trainer.evaluate()
    if canonical(trainer.checkpoint()) != frozen:
        raise RuntimeError("evaluation mutated loaded model, optimizer, RNG or counters")
    if checkpoint.read_bytes() != original_bytes:
        raise RuntimeError("source checkpoint changed during evaluation")
    return {"schema_version": 1, "evaluation_type": "fresh_frozen_neural_vs_uniform_random",
            "source_checkpoint_sha256": hashlib.sha256(original_bytes).hexdigest(),
            "source_checkpoint_bytes": len(original_bytes), "snapshot": trainer.manifest,
            "observation_schema": trainer.observation_schema,
            "episode_config": trainer.episode_config, "policy_config": trainer.policy.config,
            "training_seed_interval": {"start_inclusive": start, "end_exclusive": end},
            "training_episodes_attempted": trainer.episodes_attempted,
            "training_episodes_completed": trainer.episodes_completed,
            "optimizer_updates": trainer.policy.steps,
            "original_reserved_heldout_seeds": original_reservation,
            "fresh_heldout_seeds": seeds, "paired_comparison": paired_deltas(evaluation, seeds),
            "evaluation": evaluation, "elapsed_seconds": time.perf_counter() - started,
            "models_and_source_checkpoint_unchanged": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--seed-start", required=True, type=int)
    parser.add_argument("--seed-count", required=True, type=int)
    parser.add_argument("--report", required=True, type=Path, help="new JSON report, never overwritten")
    args = parser.parse_args(argv)
    if args.report.exists() or args.report.resolve() == args.checkpoint.resolve():
        parser.error("choose a new report path distinct from checkpoint")
    try:
        report = evaluate_checkpoint(args.checkpoint, args.snapshot,
                                     seed_start=args.seed_start, seed_count=args.seed_count)
        with args.report.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))
    comparison = {k: v for k, v in report["paired_comparison"].items() if k != "pairs"}
    print(json.dumps({"paired_comparison": comparison,
                      "initial": report["evaluation"]["initial"]["summary"],
                      "trained": report["evaluation"]["trained"]["summary"],
                      "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
