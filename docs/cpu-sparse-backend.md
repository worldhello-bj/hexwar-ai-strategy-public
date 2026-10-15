# Optional SciPy CSR CPU backend

Default execution remains NumPy. Request `--projection-backend scipy_csr` through `tools/cloud_train.py`, `env.neural_train`, or the `projection_backend` training JSON option. The choice is stored in policy checkpoints and checked during resume. SciPy is imported only for the explicit CSR choice; there is no silent fallback. Optional dependencies are pinned in `requirements-sparse.txt`; SciPy 1.17.0 was already installed for the experiment.

CSR performs candidate feature projection as sparse × dense and action-gradient accumulation as sparse-transpose × dense. It removes the temporary `nonzero_features × hidden_width` gather/product and thousands of Python/scatter calls. Matrix construction is included in all measurements. Float64 is preserved. Observation gradient accumulation remains unchanged.

## Measured isolated comparison

Source commit 59009d5; simulator v8; competition 7527; schema 2, 4096 buckets, 64 hidden units, four PPO epochs, BLAS one thread. Exactly one full 1800-simulated-second rollout was collected, then its 14 frozen learner decisions / 7,503 candidates were reused by each backend. No new game collection occurred during timing. Two trials ran in reverse backend order while other independent training workers were active; absolute times depend on that load.

- NumPy four-epoch update: 10.145 s and 9.464 s
- CSR four-epoch update: 1.304 s and 1.230 s
- Update speedup: 7.78× and 7.69×
- NumPy forward over all decisions: 0.477–0.534 s
- CSR forward over all decisions: 0.028–0.038 s
- NumPy process peak RSS: 148.7–148.9 MiB
- CSR process peak RSS: 130.2–130.3 MiB, including SciPy imports

These measure encoded-input inference/optimization, not full training throughput. Hash encoding, simulation, checkpoint JSON serialization and evaluation orchestration are unchanged. Full-game/long-run effects must be checked in a subsequent controlled branch.

## Numerical behavior

This backend does not promise bit-identical results to NumPy. Accumulation order changes slightly. On the frozen real trajectory:

- Initial probabilities/value predictions: exactly equal
- Maximum gradient difference: 1.1724e-13
- Maximum parameter difference after four Adam steps: 1.3073e-10, in a policy component whose tiny gradient is amplified by Adam's denominator
- Maximum post-update action probability difference: 4.3368e-18
- RNG state, scalar loss and reported gradient norm: exactly equal

An initial 1e-11 absolute tensor threshold intentionally exposed the parameter difference. Cross-backend evidence uses an explicit 1e-9 absolute / 1e-10 relative bound; existing gradient and legacy compatibility assertions were not relaxed. Same-backend save/load/resume remains exact in the tested runtime. Long stochastic trajectories can diverge; do not mix a backend change into an already-running experiment.

## Explicit migration, only after a run is stopped and reviewed

`python tools/migrate_neural_backend.py --source OLD.json --destination NEW.json --snapshot scenario/7527 --allow-real-map`

This creates NEW.json and NEW.json.migration.json. It never overwrites or modifies OLD.json. Preconditions: valid current scoring/snapshot contract, positive completed episode count, and an empty pending batch. Only three backend configuration fields change, including the frozen initial model's backend declaration. Parameters, optimizer moments, step counts, return statistics, RNG state, game settings and episode counters are preserved. The sidecar records parent/output SHA-256 and state fingerprints.

No real training checkpoint was converted during this investigation. Start a separate continuation branch; preserve the parent run and migration audit. The general CLI can continue a migrated checkpoint into new outputs with `python -m env.neural_train --resume NEW.json --snapshot scenario/7527 --allow-real-map --episodes ADDITIONAL_EPISODES --checkpoint NEXT.json --report REPORT.json`. Its episode budget is additional attempts. For a monitored continuation, use `python tools/cloud_train.py --initialize-from NEW.json --parent-training-report PARENT/training.json --output NEW_OUTPUT --run-id NEW_RUN --projection-backend scipy_csr --episodes CUMULATIVE_TARGET` and explicitly match the parent's seed, train-seed start, observation schema, batch size, scoring mode and held-out seeds. `--episodes` here is the total cumulative target, not an additional budget.

Initialization requires a new target run directory, empty pending batch, current v8 scoring/snapshot, matching full episode/policy configuration, and an exact-length parent journal. The journal's counters, seeds/seats, scoring versions, terminal results, summary and elapsed time are validated. Its original rows are copied unchanged. `import-audit.json` and the new manifest record parent checkpoint/report SHA-256, original counts, and any backend-migration audit. Parent files and manifests are never changed. New checkpoint serialization preserves loaded arrays, optimizer and RNG; unit tests compare both initialization and subsequent same-backend continuation exactly. No real checkpoint import was performed during development.

An interrupted imported run uses ordinary `--resume` with the same new-run settings; omit `--initialize-from` and `--parent-training-report`. Source/config compatibility checks remain active and are never bypassed.

## Reproduction

In the isolated prototype, run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark_sparse.py collect` once to save a frozen rollout, then run the same command with `numpy` and `scipy_csr`. Results and numerical tensor comparisons are under `perf-results/`. Only locally generated trusted pickle/NumPy artifacts are loaded.

## Test dependencies

The default CI installs only `requirements-neural.txt`. CSR execution, migration, and imported-continuation tests explicitly skip when SciPy is absent; NumPy compatibility and CLI opt-in tests still run. For the full optional suite:

```bash
python -m pip install -r requirements-neural.txt -r requirements-sparse.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```
