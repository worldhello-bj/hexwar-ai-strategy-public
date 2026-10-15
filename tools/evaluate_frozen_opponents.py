#!/usr/bin/env python3
"""Paired, evaluation-only neural checkpoint matches on synthetic fixtures."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env.episode_api import TwoPlayerEpisode
from env.match_eval import SIDES, _positive_int, snapshot_manifest
from env.neural_policy import NeuralPolicy
from env.neural_train import NeuralTrainer, canonical, derived_seed
from env.public_observation import PublicObservationAdapter
from env.spatial_observation import SpatialObservationAdapter
from env.contact_observation import ContactObservationAdapter


def state_hash(policy):
    return hashlib.sha256(canonical(policy.checkpoint()).encode()).hexdigest()


class FrozenSampler:
    """Accept only detached public inputs; never sample the model's training RNG."""
    def __init__(self, policy, seed):
        self.policy = policy
        self.rng = np.random.default_rng(seed)

    def __call__(self, observation, candidates):
        probs = np.asarray(self.policy.probabilities(copy.deepcopy(observation), copy.deepcopy(candidates)))
        if (probs.shape != (len(candidates),) or not np.isfinite(probs).all()
                or (probs < 0).any() or not np.isclose(probs.sum(), 1)):
            raise ValueError("invalid legal-action probabilities")
        return copy.deepcopy(candidates[int(self.rng.choice(len(candidates), p=probs))])


def run_game(snapshot, policies, schemas, config, *, seed, a_side):
    episode = TwoPlayerEpisode(snapshot, **config)
    public = {2: PublicObservationAdapter(episode), 3: SpatialObservationAdapter(episode),
              4: ContactObservationAdapter(episode)}
    seats = {side: "A" if side == a_side else "B" for side in SIDES}
    samplers = {side: FrozenSampler(policies[seats[side]], derived_seed("frozen_match", seed, side, seats[side]))
                for side in SIDES}
    status = episode.reset(seed=seed)
    trace_hash = hashlib.sha256()
    while not status["terminated"] and not status["truncated"]:
        side = status["current_actor"]
        schema = schemas[seats[side]]
        observation = (public[schema] if schema in (2, 3, 4) else episode).observe(side)
        candidates = public[schema].candidate_views(side) if schema in (2, 3, 4) else episode.legal_actions(side)
        if not candidates:
            status = {**status, "truncated": True, "reason": "no_legal_actions", "winner": None,
                      "final_scores": None, "rewards": {s: 0 for s in SIDES}}
            break
        choice = samplers[side](observation, candidates)
        if canonical(choice) not in {canonical(candidate) for candidate in candidates}:
            raise RuntimeError("sampler returned unadvertised action")
        action = choice["action"] if schema in (2, 3, 4) else choice
        trace_hash.update(canonical([side, observation, candidates, action]).encode())
        status = episode.step(action, side=side)
        if not status["terminated"] and any(status["rewards"].values()):
            raise RuntimeError("nonterminal reward")
    scores = status["final_scores"] if status["terminated"] and not status["truncated"] else None
    other = next(side for side in SIDES if side != a_side)
    difference = scores[a_side] - scores[other] if scores is not None else None
    return {"seed": seed, "a_side": a_side, "seats": seats, "status": status,
            "raw_terminal_scores": scores,
            "a_score": scores[a_side] if scores is not None else None,
            "b_score": scores[other] if scores is not None else None,
            "a_score_diff": difference,
            "result": None if difference is None else "win" if difference > 0 else "loss" if difference < 0 else "draw",
            "public_trace_sha256": trace_hash.hexdigest()}


def summarize(games):
    completed = [g for g in games if g["result"] is not None]
    return {"games": len(games), "completed": len(completed), "truncated": len(games) - len(completed),
            **{name: sum(g["result"] == name for g in completed) for name in ("win", "draw", "loss")},
            **{name: statistics.mean(g[name] for g in completed) if completed else None
               for name in ("a_score", "b_score", "a_score_diff")}}


def evaluate(checkpoint_a, checkpoint_b, snapshot, *, seeds, b_initial=False):
    seeds = list(seeds)
    if not seeds or len(seeds) > 256 or len(set(seeds)) != len(seeds):
        raise ValueError("provide 1 to 256 unique fresh seeds")
    for seed in seeds:
        _positive_int("seed", seed, True)
    paths = {"A": Path(checkpoint_a), "B": Path(checkpoint_b)}
    sources = {label: path.read_bytes() for label, path in paths.items()}
    trainers = {label: NeuralTrainer.load(path, snapshot) for label, path in paths.items()}
    a, b = trainers.values()
    if a.episode_config != b.episode_config:
        raise ValueError("checkpoint episode contracts differ")
    for trainer in trainers.values():
        start = trainer.train_seed_start
        if any(start <= seed < start + trainer.episodes_attempted or seed in trainer.heldout_seeds for seed in seeds):
            raise ValueError("evaluation seeds overlap training or reserved heldout seeds")
    models = {label: NeuralPolicy.from_checkpoint(trainer.initial_policy if label == "B" and b_initial
                                                 else trainer.policy.checkpoint())
              for label, trainer in trainers.items()}
    frozen = {label: state_hash(model) for label, model in models.items()}
    schemas = {label: trainer.observation_schema for label, trainer in trainers.items()}
    manifest = snapshot_manifest(snapshot)
    games = []
    for seed in seeds:
        for side in SIDES:
            games.append(run_game(snapshot, models, schemas, a.episode_config, seed=seed, a_side=side))
            if snapshot_manifest(snapshot) != manifest:
                raise RuntimeError("snapshot changed during evaluation")
            if any(state_hash(model) != frozen[label] for label, model in models.items()):
                raise RuntimeError("frozen model, optimizer or training RNG changed")
    if any(path.read_bytes() != sources[label] for label, path in paths.items()):
        raise RuntimeError("source checkpoint changed")
    pairs = []
    for seed in seeds:
        rows = [g for g in games if g["seed"] == seed]
        complete = all(g["result"] is not None for g in rows)
        pairs.append({"seed": seed, "complete": complete,
                      "mean_a_score_diff": statistics.mean(g["a_score_diff"] for g in rows) if complete else None})
    values = [p["mean_a_score_diff"] for p in pairs if p["complete"]]
    return {"schema_version": 1, "evaluation_type": "frozen_neural_head_to_head", "snapshot": manifest,
            "episode_config": a.episode_config, "observation_schemas": schemas, "seeds": seeds,
            "models": {label: {"source_sha256": hashlib.sha256(sources[label]).hexdigest(),
                               "state_sha256": frozen[label], "initial": label == "B" and b_initial,
                               "training_episodes_attempted": trainers[label].episodes_attempted}
                       for label in models},
            "summary": summarize(games), "by_a_seat": {side: summarize([g for g in games if g["a_side"] == side]) for side in SIDES},
            "paired": {"complete_pairs": len(values), "excluded_pairs": len(seeds) - len(values),
                       "mean_a_score_diff": statistics.mean(values) if values else None,
                       "sample_standard_error": statistics.stdev(values) / len(values)**0.5 if len(values) > 1 else None,
                       "pairs": pairs}, "games": games, "frozen_state_and_source_verified": True,
            "caveat": "Synthetic game only. W/D/L compares raw terminal scores, not tactical rewards. Seat swaps share scenario seeds; policy RNG streams are independent by seat and label. Self-play is symmetric in expectation, not necessarily exact finite-sample wins. Truncations are excluded. Standard error is descriptive over complete seed pairs, not proof of improvement."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", required=True, type=Path)
    parser.add_argument("--checkpoint-b", required=True, type=Path)
    parser.add_argument("--b-initial", action="store_true")
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--seed-start", required=True, type=int)
    parser.add_argument("--seed-count", required=True, type=int)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        _positive_int("seed_count", args.seed_count)
        if args.seed_count > 256:
            raise ValueError("seed_count exceeds 256")
        if args.report.exists() or args.report.resolve() in (args.checkpoint_a.resolve(), args.checkpoint_b.resolve()):
            raise ValueError("choose a new report path distinct from checkpoints")
        report = evaluate(args.checkpoint_a, args.checkpoint_b, args.snapshot,
                          seeds=range(args.seed_start, args.seed_start + args.seed_count), b_initial=args.b_initial)
        with args.report.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
