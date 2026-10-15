#!/usr/bin/env python3
"""Export one frozen learned-policy trajectory, never a full policy enumeration.

Uses exactly NeuralTrainer's evaluation/opponent sampling seeds and conventions.
Only learner-seat observations and its legal candidates are exported. Public
score changes are review markers, not causal explanations or reward shaping.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.episode_api import TwoPlayerEpisode
from env.match_eval import SIDES, _positive_int, snapshot_manifest
from env.neural_train import NeuralTrainer, canonical, derived_seed
from env.public_observation import PublicObservationAdapter


def export_replay(checkpoint, snapshot, *, seed, learner_side):
    _positive_int("seed", seed, True)
    if learner_side not in SIDES:
        raise ValueError("learner_side must be BLUE or RED")
    checkpoint = Path(checkpoint)
    source = checkpoint.read_bytes()
    trainer = NeuralTrainer.load(checkpoint, snapshot)
    frozen = canonical(trainer.checkpoint())
    episode = TwoPlayerEpisode(snapshot, **trainer.episode_config)
    view = PublicObservationAdapter(episode) if trainer.observation_schema == 2 else episode
    status = episode.reset(seed=seed)
    opponent_rng = random.Random(derived_seed(trainer.seed, seed, learner_side, "opponent"))
    eval_rng = np.random.default_rng(derived_seed(trainer.seed, seed, learner_side, "evaluation"))
    decisions, review_points = [], []
    initial_observation = view.observe(learner_side)
    previous_score = initial_observation.get("score")
    while not status["terminated"] and not status["truncated"]:
        side = status["current_actor"]
        candidates = (view.candidate_views(side)
                      if side == learner_side and trainer.observation_schema == 2
                      else episode.legal_actions(side))
        if not candidates:
            status = {**status, "terminated": False, "truncated": True,
                      "reason": "no_legal_actions", "final_scores": None}
            break
        if side == learner_side:
            observation = view.observe(side)
            probabilities = trainer.policy.probabilities(observation, candidates)
            selected = int(eval_rng.choice(len(candidates), p=probabilities))
            original_actions = ([candidate["action"] for candidate in candidates]
                                if trainer.observation_schema == 2 else candidates)
            action = original_actions[selected]
            decisions.append(copy.deepcopy({
                "decision_index": len(decisions), "step": status["steps"],
                "game_time": observation["time"], "observation": observation,
                "candidates": [{"index": i, "action": original_actions[i],
                                "probability": float(p),
                                **({"public_context": candidates[i]["public_context"]}
                                   if trainer.observation_schema == 2 else {})}
                               for i, p in enumerate(probabilities)],
                "selected_index": selected, "selected_action": action,
                "selected_probability": float(probabilities[selected]),
            }))
        else:
            action = opponent_rng.choice(candidates)
        status = episode.step(action, side=side)
        # Observe only the learner seat, even following an opponent action.
        after = view.observe(learner_side)
        score = after.get("score")
        if score is not None and score != previous_score:
            review_points.append({"kind": "public_score_change", "after_step": status["steps"],
                                  "game_time": after["time"], "previous_score": previous_score,
                                  "score": score,
                                  "latest_learner_decision_index": len(decisions) - 1 if decisions else None})
        previous_score = score
    final_observation = view.observe(learner_side)
    completed = status["terminated"] and not status["truncated"]
    ending = {"terminated": status["terminated"], "truncated": status["truncated"],
              "reason": status["reason"], "steps": status["steps"],
              "game_time": final_observation["time"],
              "own_original_score": status["final_scores"][learner_side] if completed else None}
    if "score" in final_observation:
        ending["public_score"] = final_observation["score"]
    review_points.append({"kind": "game_end" if completed else "truncation", **ending})
    if canonical(trainer.checkpoint()) != frozen:
        raise RuntimeError("replay mutated loaded parameters, optimizer, RNG or counters")
    if checkpoint.read_bytes() != source:
        raise RuntimeError("source checkpoint changed during replay")
    if snapshot_manifest(snapshot) != trainer.manifest:
        raise RuntimeError("synthetic snapshot changed during replay")
    report = {"schema_version": 1, "replay_type": "frozen_learned_policy_vs_uniform_random",
            "seed": seed, "learner_side": learner_side,
            "source_checkpoint_sha256": hashlib.sha256(source).hexdigest(),
            "snapshot_sha256": trainer.manifest["sha256"],
            "observation_schema": trainer.observation_schema,
            "episode_config": trainer.episode_config,
            "policy_config": trainer.policy.config,
            "training_episodes_completed": trainer.episodes_completed,
            "initial_observation": initial_observation, "decisions": decisions,
            "review_points": review_points, "final_observation": final_observation,
            "ending": ending, "models_and_source_checkpoint_unchanged": True,
            "caveat": "One sampled trajectory of the learned stochastic policy, not a complete strategy table, a win claim, or preset tactics. Review markers show observed changes only; they do not identify causes. Opponent actions and private state are not exported. Truncation is not a scored result."}


    if trainer.observation_schema == 2:
        report["public_board"] = copy.deepcopy(initial_observation["board"])
        for observation in [initial_observation, final_observation,
                            *[d["observation"] for d in decisions]]:
            if observation.pop("board") != report["public_board"]:
                raise RuntimeError("static public board changed during replay")
            observation["board_ref"] = "#/public_board"
    return report


def reconstruct_observation(report, observation):
    """Restore the exact policy input from a recorded observation."""
    result = copy.deepcopy(observation)
    if "board_ref" in result:
        if result.pop("board_ref") != "#/public_board":
            raise ValueError("unsupported public board reference")
        result["board"] = copy.deepcopy(report["public_board"])
    return result


def write_csv(report, stream):
    """One row per legal candidate; selected rows mark the original action taken."""
    columns = ["decision_index", "step", "game_time", "own_public_score", "opponent_public_total",
               "candidate_index", "probability", "selected", "original_action"]
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for decision in report["decisions"]:
        score = decision["observation"].get("score", {})
        for candidate in decision["candidates"]:
            writer.writerow({**{key: decision[key] for key in ("decision_index", "step", "game_time")},
                             "own_public_score": score.get("own", ""),
                             "opponent_public_total": score.get("enemy", ""),
                             "candidate_index": candidate["index"], "probability": candidate["probability"],
                             "selected": candidate["index"] == decision["selected_index"],
                             "original_action": canonical(candidate["action"])})


def write_markdown(report, stream):
    """Concise selected choices; full probabilities and observations stay in JSON."""
    stream.write("# Observed learned choices\n\n")
    stream.write(f"Seed {report['seed']}; learner {report['learner_side']}; frozen checkpoint "
                 f"`{report['source_checkpoint_sha256']}`.\n\n")
    stream.write("One sampled trajectory, not a complete policy or an explanation of its reasoning. "
                 "No preset tactics or causal explanations are supplied. Full candidate probabilities "
                 "and reconstructible observations are in the companion JSON.\n\n")
    stream.write("| Decision | Step | Time | Own / opponent public totals | Selected probability | Original selected action |\n")
    stream.write("|---:|---:|---:|---|---:|---|\n")
    for decision in report['decisions']:
        score = decision['observation'].get('score')
        totals = f"{score['own']} / {score['enemy']}" if score is not None else "not exposed"
        action = canonical(decision['selected_action']).replace('|', '&#124;')
        stream.write(f"| {decision['decision_index']} | {decision['step']} | {decision['game_time']} | "
                     f"{totals} | {decision['selected_probability']:.8f} | `{action}` |\n")
    stream.write("\n## Review points\n\nThese mark observable changes only, not causes or optimal decisions.\n\n")
    for point in report['review_points']:
        if point['kind'] == 'public_score_change':
            stream.write(f"- Time {point['game_time']}, after step {point['after_step']}: public totals "
                         f"{canonical(point['previous_score'])} → {canonical(point['score'])}.\n")
        else:
            stream.write(f"- {point['kind']} at time {point['game_time']}: {point['reason']}; "
                         f"original own terminal score: {point['own_original_score']}.\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--learner-side", required=True, choices=SIDES)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", type=Path, help="optional concise selected-decision table")
    args = parser.parse_args(argv)
    outputs = [args.report, args.csv] + ([args.markdown] if args.markdown else [])
    if any(path.exists() for path in outputs) or len({p.resolve() for p in outputs}) != len(outputs):
        parser.error("choose distinct new JSON and CSV paths; existing files are never overwritten")
    try:
        report = export_replay(args.checkpoint, args.snapshot, seed=args.seed, learner_side=args.learner_side)
        with args.report.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        with args.csv.open("x", newline="", encoding="utf-8") as stream:
            write_csv(report, stream)
        if args.markdown:
            with args.markdown.open("x", encoding="utf-8") as stream:
                write_markdown(report, stream)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"seed": args.seed, "learner_side": args.learner_side,
                      "decisions": len(report["decisions"]), "ending": report["ending"],
                      "review_points": len(report["review_points"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
