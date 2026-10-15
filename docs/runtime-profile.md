# Offline synthetic API runtime profile

Measured 2026-10-14 in the current cloud workspace; no network, training, tactical optimization, or external data. No engine/environment implementation was changed.

## Reproduction and workload

```sh
timeout 110s python tools/benchmark_contracts.py --output /tmp/hexwar-runtime-profile.json
```

The script reuses `EnvContractTests.setUp` from `tests/test_env_contract.py`, and asserts that every entity has an empty weapon list. This original fixture has 50 map cells, two abstract entities (`data=0`), and one synthetic objective. Fog is enabled; the API's default move radius is 6. It does not load the demo scenario. Actions are uniformly sampled from the legal-action list, with a fixed selection seed. No policy is learned or optimized. Reset timing invokes the API's ordinary default seed; the evolving step workload starts with seed 17.

Timing and allocation tracing are separate runs. CPU uses `time.process_time`; wall time uses `time.perf_counter`. Step timing excludes action enumeration and episode reset. Throughput includes reset, action enumeration, selection, step, spawn startup, and teardown. Each process-count trial performs the same two seeded jobs, each with three episodes of eight decisions (48 decisions total). The one-process trial runs first. Total benchmark wall time: **3.667 seconds**.

## Environment

- Python 3.12.14, Clang 22.1.3
- Linux 6.18.44 x86_64, glibc 2.41
- Host-reported CPU: AMD EPYC 9V74 80-Core Processor
- `os.cpu_count()` and process CPU affinity both report 9 CPUs. This is not evidence of nine dedicated cores or guaranteed CPU quota.
- Multiprocessing start method: `spawn`
- Base checkout: `4311f7625f728611f3d3b7619b998fd1dc427437`
- The usual cgroup v2 `cpu.max` and `memory.max` files were absent; effective resource quotas are unknown.

## API timing

Milliseconds, unprofiled and without tracemalloc enabled:

| API | Calls | CPU median | CPU p95 | Wall median | Wall p95 |
|---|---:|---:|---:|---:|---:|
| reset | 100 | 0.0783 | 0.1788 | 0.0780 | 0.1781 |
| observe | 500 | 0.0072 | 0.0103 | 0.0069 | 0.0099 |
| legal_actions | 40 | 12.0651 | 14.1599 | 12.0687 | 14.1570 |
| step | 80 | 0.2991 | 0.4291 | 0.2972 | 0.4245 |

The percentile implementation selects sorted index `int(n * .95)` (zero based, capped at n−1); these are descriptive small-sample percentiles, not confidence bounds. Timer overhead matters for the very short `observe` call. Reset includes reading the fixture JSON from the filesystem; the OS cache is not flushed.

## Allocation and resident memory

Separate single-call tracemalloc samples after resetting the environment (bytes). Returned values are not retained by the benchmark. These values measure traced Python allocations during each call, not the complete environment footprint, native allocations, or per-episode growth.

| API | Traced peak bytes | Traced bytes remaining after call |
|---|---:|---:|
| reset | 18,450 | 8,180 |
| observe | 1,325 | 56 |
| legal_actions | 44,424 | 3,352 |
| step | 34,472 | 3,528 |

Parent process lifetime peak RSS was 19,024 KiB (~18.58 MiB), including imports and profiling. Each worker reported lifetime peak RSS of 18,896 KiB (~18.45 MiB). Linux `ru_maxrss` is a process high-water mark; worker peaks must not be summed as an accurate concurrent physical-memory measurement (shared pages and timing differ). This short run is not a memory-leak or long-history test.

## One versus two worker processes

| Workers | Decisions | Wall including spawn/teardown | Decisions/s | Summed job CPU |
|---|---:|---:|---:|---:|
| 1 | 48 | 0.6860 s | 69.98 | 0.5896 s |
| 2 | 48 | 0.4705 s | 102.02 | 0.7098 s |

Observed short-run speedup: **1.46×**. These are single paired trials in a shared cloud environment; startup, scheduling, CPU contention, and cache effects are material. This does not establish steady-state throughput, ideal scaling, or performance on the user's machine. It does not measure parallel state sharing or interprocess observation transfer.

## Measured bottleneck

A separate cProfile run over 15 `legal_actions` calls took 0.920 s (profiling adds substantial overhead, so this duration must not replace the unprofiled latency figures).

- `legal_actions`: 0.919 s cumulative
- `_try_actions`: 750 calls, 0.893 s cumulative
- `copy.deepcopy`: 750 top-level invocations, 618,750 total recursive calls, 0.809 s cumulative
- `_apply`: 735 calls, 0.057 s cumulative
- `step_seconds`: 3,540 calls, 0.052 s cumulative

Approximately **88% of profiled legal-action elapsed time** is attributed to deepcopy. These cumulative figures overlap and must not be added. Every candidate is validated by `_try_actions(commit=False)`, which clones mutable game state and its random stream. The map dictionary is already memo-shared rather than deep-copied. At the initial synthetic state, 50 candidate transactions are attempted per legal-action enumeration, which explains why enumeration is much more expensive than a single step.

This supports a software-engineering investigation of transaction-validation/state-copy overhead, not a gameplay conclusion. Any future optimization must preserve failure atomicity, public object identities, observation isolation, and RNG state equivalence. No such optimization was implemented or validated here.

## Limits and verification

This is a deliberately tiny, unarmed synthetic contract workload. It cannot characterize large maps/entity counts, long histories, real scenarios, combat branches, model inference, GPU training, or training sample efficiency. The result says nothing about tactical performance or real-world operations. The script guards the fixture against accidental armed input and uses only standard-library dependencies.

Verification: the full benchmark completed successfully with both spawned worker counts; Python bytecode compilation passed. Existing engine/env regression tests were not rerun for this profiling-only addition. Detailed raw results are regenerated by the command above, including the top 20 cumulative cProfile entries.
