#!/usr/bin/env python3
"""Bounded, offline API microbenchmark on the existing unarmed test fixture.

No policy training or tactical optimization. Only uniformly random legal actions.
Run from any directory: python tools/benchmark_contracts.py --output /tmp/profile.json
"""
import argparse
import cProfile
import importlib.util
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import pstats
import random
import resource
import statistics
import sys
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.hexwar_env import HexwarEnv


def fixture():
    spec = importlib.util.spec_from_file_location("contract_fixture", ROOT / "tests/test_env_contract.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    case = module.EnvContractTests()
    case.setUp()
    assert all(not u.wz for u in case.env.game.units.values()), "Only unarmed fixtures allowed"
    return case


def summary(values):
    values = sorted(values)
    return {"n": len(values), "median_ms": statistics.median(values) * 1000,
            "p95_ms": values[min(len(values)-1, int(len(values)*.95))] * 1000,
            "mean_ms": statistics.mean(values) * 1000}


def measurement(env, operation, n, rng):
    cpu, wall = [], []
    for _ in range(n):
        if env.done:
            env.reset(seed=17)
        action = rng.choice(env.legal_actions()) if operation == "step" else None
        fn = (lambda: env.step(action)) if operation == "step" else getattr(env, operation)
        c0, w0 = time.process_time(), time.perf_counter()
        fn()
        wall.append(time.perf_counter() - w0)
        cpu.append(time.process_time() - c0)
    return {"cpu": summary(cpu), "wall": summary(wall)}


def throughput(job):
    path, seed = job
    env = HexwarEnv(snapshot=path)
    rng = random.Random(seed)
    steps = 0
    c0 = time.process_time()
    for episode in range(3):
        env.reset(seed=seed + episode)
        for _ in range(8):
            if env.done:
                break
            env.step(rng.choice(env.legal_actions()))
            steps += 1
    return {"steps": steps, "cpu_s": time.process_time()-c0,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/hexwar-runtime-profile.json")
    args = parser.parse_args()
    start = time.perf_counter()
    case = fixture()
    try:
        result = {"environment": {"python": sys.version, "platform": platform.platform(),
                  "cpu_count": os.cpu_count(), "affinity_cpus": len(os.sched_getaffinity(0)),
                  "multiprocessing": "spawn", "cwd": str(ROOT)},
                  "fixture": {"source": "tests/test_env_contract.py:EnvContractTests.setUp",
                  "hexes": len(case.env.game.hexes), "entities": len(case.env.game.units),
                  "weapons": 0, "fog": True, "move_radius": 6}, "operations": {}}
        for op, count in [("reset",100), ("observe",500), ("legal_actions",40), ("step",80)]:
            env = HexwarEnv(snapshot=str(case.root))
            env.reset(seed=17)
            result["operations"][op] = measurement(env, op, count, random.Random(17))
            env.reset(seed=17)
            action = random.Random(17).choice(env.legal_actions()) if op == "step" else None
            tracemalloc.start()
            if op == "step":
                env.step(action)
            else:
                getattr(env, op)()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result["operations"][op]["single_call_alloc_bytes"] = {"retained": current, "peak": peak}
        result["throughput"] = {}
        for workers in (1, 2):
            w0 = time.perf_counter()
            with mp.get_context("spawn").Pool(workers) as pool:
                jobs = pool.map(throughput, [(str(case.root), 101), (str(case.root), 102)])
            elapsed = time.perf_counter() - w0
            steps = sum(job["steps"] for job in jobs)
            result["throughput"][str(workers)] = {"wall_s_including_spawn": elapsed,
                "steps": steps, "steps_per_s": steps/elapsed, "jobs": jobs}
        env = HexwarEnv(snapshot=str(case.root))
        env.reset(seed=17)
        profiler = cProfile.Profile()
        profiler.enable()
        measurement(env, "legal_actions", 15, random.Random(17))
        profiler.disable()
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
        stats.print_stats(20)
        result["legal_actions_cprofile"] = stream.getvalue()
        result["parent_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["total_wall_s"] = time.perf_counter()-start
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        case.doCleanups()


if __name__ == "__main__":
    main()
