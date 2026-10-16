"""Offline software evaluation on original synthetic fixtures, never training.

The scheduler is SERIAL ALTERNATING, not simultaneous: BLUE acts first and each
accepted action advances the shared clock exactly once. Each seed is run twice
with policy seats exchanged. This balances seat/first-move exposure, not arbitrary
asymmetries or action duration effects. Policies receive detached JSON only.

Policy factories must return fresh, independent callables (observation, actions).
This is an API isolation contract, not a sandbox for hostile Python code.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import random

from .hexwar_env import HexwarEnv

SIDES = ("BLUE", "RED")
FILES = ("map_hexes.json", "cities.json", "units.json")


def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_synthetic_snapshot(path):
    """Write a tiny original abstract-token software fixture, with no weapons."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    data = {
        "map_hexes.json": {str(p): {"step": 1, "cond": 0, "elevation": 0}
                           for p in (20004, 20006, 20008, 20010)},
        "cities.json": [],
        "units.json": [{"id": side.lower(), "name": "Token " + side,
                        "color": side, "typeid": 1, "pos": pos,
                        "blood": 4, "score": 1, "wz": ""}
                       for side, pos in zip(SIDES, (20004, 20010))],
        "evaluation_provenance.json": {"origin": "original_synthetic",
                                       "purpose": "software_smoke_test"},
    }
    for name, value in data.items():
        target = path / name
        if target.exists():
            raise FileExistsError(target)
    for name, value in data.items():
        (path / name).write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def snapshot_manifest(path, *, allow_real=False):
    """Require explicit synthetic provenance; hash exact consumed fixture bytes.

    A provenance declaration is an attestation, not an automated authenticity test.

    `allow_real=True` 是**显式 opt-in**，用于在**本地真实想定**（`snapshot/parsed`，
    不入库）上训练：此时来源标记为 `origin="local_real_map"`，
    仍然对三个被读取文件做哈希，所以同一次运行仍可复现。
    默认 `False` 时行为与以前完全一致：只接受 `original_synthetic`。
    """
    path = Path(path)
    prov_path = path / "evaluation_provenance.json"
    if prov_path.exists():
        provenance = json.loads(prov_path.read_text("utf-8"))
        if provenance.get("origin") != "original_synthetic":
            raise ValueError("unsupported snapshot provenance: %r"
                             % provenance.get("origin"))
    elif allow_real:
        provenance = {"origin": "local_real_map", "purpose": "training_only"}
    else:
        raise ValueError("evaluation requires an original synthetic snapshot")
    names = FILES + (("evaluation_provenance.json",) if prov_path.exists() else ())
    hashes = {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
              for name in names}
    return {"files": hashes, "sha256": _hash(hashes), "provenance": provenance}


def random_policy(seed):
    """Independent uniform legal-action sampler for software smoke tests only."""
    rng = random.Random(seed)

    def choose(observation, legal_actions):
        return copy.deepcopy(rng.choice(legal_actions))
    return choose


def _positive_int(name, value, allow_zero=False):
    if (not isinstance(value, int) or isinstance(value, bool)
            or value < (0 if allow_zero else 1)):
        raise ValueError(name + " has an invalid integer value")


class _SerialSession:
    """Own one authoritative state and keep separate per-seat observation history."""
    def __init__(self, snapshot, seed, max_seconds, fog, dt, victory,
                 scoring_mode="verified", deploy_window="opening"):
        self.env = HexwarEnv(snapshot=snapshot, side="BLUE", max_seconds=max_seconds,
                             fog=fog, dt=dt, victory=victory,
                             scoring_mode=scoring_mode,
                             deploy_window=deploy_window)
        self.env.reset(seed=seed)
        self.histories = {"BLUE": self.env._history, "RED": []}
        self.refresh()

    def select(self, side):
        self.env.side = side
        self.env._history = self.histories[side]
        self.env.done = False
        self.env._check_done()
        return self.env

    def refresh(self):
        for side in SIDES:
            self.select(side)._remember()

    def observations(self):
        return {side: _json_copy(self.select(side).observe()) for side in SIDES}

    def outcome(self):
        g = self.env.game
        alive = {s: any(u.alive() and u.color == s for u in g.units.values())
                 for s in SIDES}
        # Natural terminal causes take precedence when coincident with a limit.
        if not any(alive.values()):
            return {"terminated": True, "truncated": False,
                    "reason": "both_sides_empty", "winner": None}
        if not all(alive.values()):
            return {"terminated": True, "truncated": False,
                    "reason": "one_side_empty", "winner": next(s for s in SIDES if alive[s])}
        if self.env.victory == "capture" and g.cities:
            for side in SIDES:
                if all(c.get("flag") == side for c in g.cities):
                    return {"terminated": True, "truncated": False,
                            "reason": "all_objectives_owned", "winner": side}
        if g.second >= self.env.max_seconds:
            return _truncated("time_limit")
        return None


def _truncated(reason, **extra):
    return dict(terminated=False, truncated=True, reason=reason, winner=None, **extra)


def run_match(snapshot, policies, *, seed=0, max_seconds=120, max_turns=100,
              fog=True, dt=10, move_radius=1, victory="score"):
    """Evaluate two already independent policies; never infer wins from truncation.

    Zero-duration actions consume a turn. A turn budget bounds stalls. A callable
    that never returns needs external process supervision; this is not a timeout
    sandbox. Exceptions/rejected actions are recorded as incomplete runs, not wins.
    """
    _positive_int("seed", seed, True)
    _positive_int("max_turns", max_turns, True)
    _positive_int("move_radius", move_radius, True)
    if set(policies) != set(SIDES) or any(not callable(policies[s]) for s in SIDES):
        raise ValueError("provide one callable for BLUE and RED")
    if policies["BLUE"] is policies["RED"]:
        raise ValueError("policies must be independent instances")
    manifest = snapshot_manifest(snapshot)
    session = _SerialSession(snapshot, seed, max_seconds, fog, dt, victory)
    if snapshot_manifest(snapshot) != manifest:
        raise ValueError("snapshot changed while initializing match")
    initial = session.observations()
    trace = []
    outcome = session.outcome()
    for turn in range(max_turns):
        if outcome:
            break
        side = SIDES[turn % 2]
        env = session.select(side)
        obs = _json_copy(env.observe())
        legal = _json_copy(env.legal_actions(move_radius=move_radius))
        row = {"turn": turn, "side": side, "time_before": env.game.second,
               "observation": obs, "observation_sha256": _hash(obs),
               "legal_actions": legal, "legal_actions_sha256": _hash(legal)}
        trace.append(row)
        if not legal:
            outcome = _truncated("no_legal_actions", side=side)
            break
        try:
            action = _json_copy(policies[side](_json_copy(obs), _json_copy(legal)))
        except Exception as exc:
            row["error"] = {"type": type(exc).__name__, "message": str(exc)}
            outcome = _truncated("policy_exception", side=side)
            break
        row["action"] = action
        if action not in legal:
            outcome = _truncated("policy_action_not_legal", side=side)
            break
        _, reward, _, info = env.step(action)
        row.update(time_after=env.game.second, reward=reward, info=_json_copy(info))
        session.refresh()
        row["state_observations_sha256"] = _hash(session.observations())
        if info.get("invalid_action"):
            outcome = _truncated("environment_rejected_legal_action", side=side)
            break
        outcome = session.outcome()
    if outcome is None:
        outcome = _truncated("turn_limit")
    final = session.observations()
    # Observation.done uses the single-agent API; match outcome is authoritative.
    result = {"schema_version": 1, "scheduler": "serial_alternating_blue_first",
              "seed": seed, "snapshot": manifest, "initial_observations": initial,
              "configuration": {"max_seconds": max_seconds, "max_turns": max_turns,
                                "fog": fog, "dt": dt, "move_radius": move_radius,
                                "victory": victory},
              "outcome": outcome, "turns": len(trace), "seconds": session.env.game.second,
              "trace": trace, "final_observations": final}
    result["record_sha256"] = _hash(result)
    return result


def summarize(records):
    """W/D/L excludes incomplete runs; CI clusters the two swapped seats per seed.

    Conservative Hoeffding 95% bounds are on A's paired mean win fraction and
    score fraction (win=1, draw=0.5). Interpretation requires independent random
    seed pairs from a target distribution, not cherry-picked deterministic seeds.
    """
    counts = {"wins": 0, "draws": 0, "losses": 0, "incomplete": 0}
    pairs = {}
    for record in records:
        outcome = record["outcome"]
        if not outcome["terminated"]:
            counts["incomplete"] += 1
            continue
        winner = outcome["winner"]
        won = winner is not None and record["seats"][winner] == "A"
        draw = winner is None
        counts["wins" if won else "draws" if draw else "losses"] += 1
        pairs.setdefault(record["pair_id"], []).append((float(won), 0.5 if draw else float(won)))
    complete = [v for v in pairs.values() if len(v) == 2]
    n = len(complete)
    intervals = {}
    for index, name in enumerate(("win_fraction", "score_fraction")):
        mean = sum(sum(x[index] for x in pair) / 2 for pair in complete) / n if n else None
        radius = math.sqrt(math.log(40) / (2 * n)) if n else None
        intervals[name] = {"mean": mean, "confidence_interval_95":
                           [max(0.0, mean - radius), min(1.0, mean + radius)] if n else None}
    return {**counts, "complete_pairs": n, "paired_statistics": intervals,
            "interval_method": "Hoeffding bound over independent seed pairs",
            "caveat": "Fixed-seed smoke tests are descriptive; no population inference is justified without a seed sampling design."}


def evaluate_paired(snapshot, factory_a, factory_b, *, seeds=(0,), **kwargs):
    """Run fresh policies for both seat assignments using matched RNG seeds."""
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("provide nonempty unique seeds")
    for seed in seeds:
        _positive_int("seed", seed, True)
    records = []
    manifest = snapshot_manifest(snapshot)
    for pair_id, seed in enumerate(seeds):
        for swapped in (False, True):
            a, b = factory_a(seed), factory_b(seed)
            seats = {"BLUE": "B" if swapped else "A", "RED": "A" if swapped else "B"}
            policies = {side: a if name == "A" else b for side, name in seats.items()}
            record = run_match(snapshot, policies, seed=seed, **kwargs)
            if record["snapshot"] != manifest:
                raise ValueError("snapshot changed during paired evaluation")
            record.update(pair_id=pair_id, seats=seats)
            record.pop("record_sha256")
            record["record_sha256"] = _hash(record)
            records.append(record)
    return {"schema_version": 1, "records": records, "summary": summarize(records)}


def main(argv=None):
    """CLI creates only an original synthetic fixture and evaluates random actions."""
    import argparse
    import tempfile
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new JSON report (never overwritten)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-seconds", type=int, default=60)
    parser.add_argument("--max-turns", type=int, default=100)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; choose a new report filename")
    with tempfile.TemporaryDirectory(prefix="synthetic-match-") as tmp:
        snapshot = write_synthetic_snapshot(Path(tmp) / "fixture")
        report = evaluate_paired(snapshot, random_policy, random_policy,
                                 seeds=args.seeds, max_seconds=args.max_seconds,
                                 max_turns=args.max_turns)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
