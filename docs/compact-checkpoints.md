# Optional compact checkpoints (isolated prototype)

Base: e1f85ee. No active run, source tree, manifest, checkpoint, or evaluator was
modified. This patch changes checkpoint transport only, not features, scoring,
optimization, seeds, or backend. The isolated cloud runner has an opt-in NPZ path; it has not been deployed or
used for a real training run.

Use a new `.npz` destination with `NeuralTrainer.save` or `NeuralPolicy.save`.
The same loaders accept JSON and NPZ based on the filename suffix. Existing
JSON checkpoint output remains JSON-compatible; the completed-160 JSON re-save
is byte-identical between the old implementation and this prototype. Renaming
files is not format conversion.

The NPZ archive contains a uint8 UTF-8 JSON metadata member and float64 model /
Adam arrays. Pending sparse decisions use packed int64 indices, float64 values and int64
ragged offsets, three arrays per decision regardless of candidate count.
All reads use `allow_pickle=False`; object dtypes are rejected before allocation.
Archive names, duplicate members/JSON keys, reference uniqueness/completeness,
NPY dtype/shape/byte lengths, finite state, configuration dimensions, and resource
bounds are checked. The transport limits metadata to 16 MiB, one array to 128 MiB,
total uncompressed entries to 512 MiB, and members to 100,000. This is not a
cryptographic authenticity scheme. Uncompressed NPZ deliberately favors fast,
bounded-memory writes over minimum file size. It is not memory-mapped.

Original trainer validation still checks scoring version/mode, snapshot hash,
algorithm/configuration, counters, baseline statistics, heldout seeds, partial
PPO epochs, pending feature bounds, and required public-score offsets. The policy
source-import guard now additionally audits the transport module's imports
against a standard-library/NumPy-only allowlist; it was not removed or relaxed
to permit engine/environment access. Source/manifest guards elsewhere were not
changed. Deployment still needs the ordinary source-provenance integration.

Initial policy state now stays in numeric arrays internally. Default checkpoint()
still returns the original detached list-based JSON tree. checkpoint(arrays=True)
is a synchronous serialization view, not a detached snapshot; callers must not
mutate it or serialize concurrently with an optimizer update. Existing JSON-only
migration tools are not generalized by this patch. Existing large JSON files
still incur JSON parsing cost on first conversion. This does not by itself make
six simultaneous whole-evaluator processes safe: rollout state and unrelated
legacy evaluator paths can still consume memory. Cloud-runner, trainer evaluation
and replay mutation checks use typed streaming SHA256 state digests without
model float lists; those checks still cover parameters, optimizer, RNG, counters
and pending decisions. The state digest includes array dtype, shape and exact
bytes and length-delimited structure. File-source and snapshot guards remain.

## Reproduction

From the isolated prototype, with a NEW output path:

```sh
OPENBLAS_NUM_THREADS=1 python benchmark_compact.py \
  analysis/compact-benchmark/completed160.json \
  --save analysis/compact-benchmark/new-conversion.npz
OPENBLAS_NUM_THREADS=1 python benchmark_compact.py \
  analysis/compact-benchmark/new-conversion.npz \
  --save analysis/compact-benchmark/new-roundtrip.npz
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -q
```

The benchmark only loads and serializes; it runs no training or evaluation.
Each invocation should run in a fresh process, sequentially. It validates the
snapshot with allow_real=True and reports process-lifetime peak RSS (Linux
ru_maxrss), wall time, file size and streaming exact state digest. Save RSS is
the peak of load plus save, not an isolated save-only incremental measurement.
The digest is calculated after timing/RSS sampling. Small synthetic regression
tests exercise updates only to verify exact resume, not learning performance.

## Completed seed7527 / 160 episodes, one copied checkpoint

Source JSON SHA256:
`df5c8f18ffd0f9a1177c63a9821a6e4c15c381fe583fc89656eb283c59df9935`.
Source and isolated copy matched after all measurements.

| Implementation / format | Load seconds | Peak load RSS MiB | Save seconds | Load+save peak MiB |
|---|---:|---:|---:|---:|
| Original e1f85ee / JSON | 4.138 | 516.8 | 4.762 | 516.8 |
| Prototype / JSON | 2.073 | 368.9 | 5.055 | 489.2 |
| Prototype / NPZ | 0.244 | 167.8 | 0.030 | 167.8 |

A final NPZ recheck after switching to the already-verified archive handle:
0.228 s, 167.8 MiB. These are single-process one-shot observations, not a rigorous
multi-run performance distribution. File sizes: source JSON 90,096,237 bytes;
canonical re-saved JSON 83,802,266 bytes; NPZ 50,364,300 bytes. Original and
prototype canonical JSON re-saves have identical SHA256
`7dd3a06f7bfe549ddfdd2a84f2f41232def67cb9f0898ec56b9999bb59189c96`.

Both formats yielded the identical full-state digest:
`c2e81c7ae97b44eb24328899eeff8cbba776d4613cd5301ad19950e2be4b165b`.
This includes policy/initial-policy parameters, Adam, RNG, config, counters and
pending data. Completed160 has no pending trajectories; partial-batch preservation
is independently covered by 12 combinations of observation schemas 2/3,
none/prior/public baselines and REINFORCE/PPO, with truncations and continuation.
JSON bytes match after NPZ roundtrips in those tests. Corruption tests reject
wrong scoring/snapshot/counters/steps, bad tensor shape/nonfinite data, object or
float32 dtype, unknown/missing/reused references, invalid metadata format,
Boolean format version, oversized configuration and huge declared NPY arrays.

Raw measurements and test logs are in `analysis/compact-benchmark/` in the
isolated working tree, excluded from the patch. No checkpoint binaries are
included in the patch.

## Real-shaped incomplete batch stress test

The completed seed7527 public replay has 17 decisions, each with 559–563
candidates. Without running games, its public observations/candidates were
rehydrated into Decision features, then copied into seven pending trajectories
(the maximum incomplete batch at batch_episodes=8). Counters were adjusted only
in this synthetic serialization fixture. This is not a trained checkpoint or a
new experimental result, and must never replace a real run checkpoint.

Packed sparse storage reduces this case from roughly 133,000 potential separate
candidate-array ZIP entries to 400 total entries (42 model arrays, 357 packed
sparse arrays, one metadata array). Metadata is 22,422 bytes; sparse feature data
98,233,744 bytes; full checkpoint 149,240,174 bytes. The 100,000-member safety cap
was not increased. Serialization took 0.226 s with 374,108,160-byte peak RSS
including replay construction and feature hashing; reload with the source still
resident took 0.622 s. Every sparse index/value, selection, score offset, policy
parameter, Adam state and RNG state matched after reload.

Reproduction: `OPENBLAS_NUM_THREADS=1 python benchmark_compact_pending.py`
requires the copied completed160.npz and the existing replay at the documented
cloud path, and intentionally refuses to overwrite its fixture destination.
A unit test also checks 600 candidates produce only three sparse-array entries
and rejects nonmonotone/out-of-range ragged offsets.


## Cloud runner integration (opt-in, new run only)

`--checkpoint-format npz` selects atomic NPZ saves and a checkpoint.npz progress
path. Default JSON behavior and old JSON manifests remain unchanged. New NPZ
manifests record checkpoint_format=npz, and resume requires the same explicit
format plus the existing exact source/config checks. NPZ saves use exclusive
UUID temporary files, fsync, os.replace under the existing per-output writer lock,
and directory fsync. Failed saves preserve the previous checkpoint and remove
only their own temporary files. Journal-first crash recovery order is unchanged.

Example command shape (not executed; choose the correct seed and new run ID):

```sh
python tools/cloud_train.py --run-id NEW_RUN --checkpoint-format npz \
  --initialize-from /path/to/completed/checkpoint.json \
  --parent-training-report /path/to/completed/training.json \
  --seed 7527 --observation-schema 3 --baseline-offset public_own_score \
  --projection-backend scipy_csr --episodes 320
```

All imported settings/counters and the parent journal must match the original
import contract. JSON and NPZ sources can initialize either JSON or NPZ new runs.
Import hashes/checks parent files, preserves historical records and writes a new
import audit; it never rewrites an old source manifest. Resume the new NPZ run
with the same options and `--resume`, omitting initialize-from/parent-report.
Stop files still take effect only at episode boundaries; incomplete batches are
saved exactly with packed sparse arrays and resumed without flushing the batch.
No real training was run to validate this integration: tests use deterministic
synthetic rollouts and actual small trainers.

A fresh process loading the seven-trajectory replay-shaped NPZ fixture used
297,959,424-byte peak RSS and 1.044 s. Loading plus saving peaked at 322,854,912
bytes; save took 0.273 s. File bytes matched at 149,240,174.

Final integrated regression run: 289 tests passed in 7.98 s, including 9 new
cloud compact tests, exact all-four-format import/resume paths, same-format
resume guards, stop-file partial batches, mutation detection, no-overwrite,
atomic replacement and failed-save preservation/cleanup. `git diff --check`
passed. No real game, training run or active-run integration was executed.
