# Optional fixed-opponent curriculum (prototype)

This is offline abstract-game infrastructure, not a result or a strength claim.
The existing uniform-legal training default remains unchanged. No training job is
started by this patch. Terminal reward is still the learner's raw own game score;
score difference and W/D/L remain diagnostics only.

## Contract

- A pool has 1–8 equally weighted entries, including at most one optional uniform
  entry. There must be at least one already-trained neural checkpoint. Checkpoint
  entries are unique, fixed NPZ files, with a combined 512 MiB on-disk limit and
  512 MiB resident policy/optimizer tensor limit. Existing NPZ archive bounds
  still apply. This is intentionally a small pool, not online population training.
- Every checkpoint must match the learner's snapshot, episode configuration,
  scoring version/mode, and public observation schema (2, 3 or 4). Model widths
  and other neural configuration may differ; the actual opponent configuration
  is recorded. JSON is not accepted for opponent sources.
- Ordered membership, SHA256, configuration, schema, scoring contract, snapshot,
  and model state digest are sealed into the trainer checkpoint and cloud run
  manifest. Paths are absolute; preserve those files for resume. Source hashes
  are checked at loading and before each episode. Changing a file, order, member,
  or uniform inclusion is a different curriculum and cannot silently resume.
- A SHA-derived episode-selection seed selects an opponent independently of
  learner sampling. Paired evaluation seats share the same selected opponent.
  Opponent actions have their own RNG. No opponent training RNG is consumed.
- Each neural seat uses a separate public observation adapter. Neither policy
  receives the opposing observation, policy weights, or environment reference.
  Opponent tensors are read-only and only `probabilities` is invoked; no scripted
  tactical actions or online weight refresh are introduced.
- Each pool rollout records opponent index, source SHA/id, kind and seat alongside
  learner seat, raw terminal scores, result and truncation status. Incomplete
  episodes remain excluded from returns and win/draw/loss summaries.
- Source models are read once through compact numeric transport; neither their
  training trajectories nor model float lists are copied into learner checkpoints.
  Resume reloads frozen policy references and restores the learner's partial batch
  and RNG exactly. Evaluation checks trainer and opponent state invariance.

## Integration

`tools/cloud_train.py` adds repeated `--opponent-checkpoint MODEL.npz` and optional
`--include-uniform-opponent`. No flags means the historical uniform default.
Use `--checkpoint-format npz` for the new run's compact checkpoint transport.
The standalone `env.neural_train` CLI is deliberately unchanged; the Python API
accepts `opponent_pool=describe_pool(paths, include_uniform=True)`.

To branch an existing completed checkpoint, choose a fresh run ID and use the
existing `--initialize-from` / `--parent-training-report` import workflow, together
with the desired pool flags and exactly matching learner configuration. Import
requires an empty partial batch. It preserves model, optimizer, RNG, episode
counters, seed continuity and all historical journal rows. The import audit and
checkpoint record the parent checkpoint SHA, previous pool, new pool and episode
at transition. This also supports an explicit new-run switch back to uniform.

For subsequent resume, pass `--resume` and the identical ordered pool flags.
Removing or changing them is rejected. Never use the live run's mutable checkpoint
as an opponent: retain versioned, immutable snapshots outside its output directory.
No migration or change is made to the existing 640-episode artifacts.

The Python `import_opponent_pool_for_new_run` method is the explicit batch-boundary
import operation, not a `load` override. The cloud runner supplies verified parent
file provenance and enforces new-run directory semantics. `NeuralTrainer.load`
has no pool-change argument.

## Recommended next experiment, not launched

Wait for the fresh broad evaluation of the existing checkpoints. Then retain both
640-episode checkpoints and journals unchanged. Branch a matched pair from the
chosen parent at its completed batch boundary:

1. Control: another 320 episode attempts against the unchanged uniform baseline.
2. Treatment: the same 320 attempts, optimizer configuration, learner seed and
   seed sequence, against an equal fixed pool of uniform plus the two retained,
   schema-compatible trained checkpoints. No latest-online opponent updates.

This is a proposal, not permission to launch. If only one compatible checkpoint
exists, use uniform plus that checkpoint and record the two-entry mixture instead.
Use a separate untouched paired-seed evaluation set after the run, against each
pool member individually and held-out trained opponents if available. Report by
seat and opponent: raw own score, raw opposing score, score difference, W/D/L,
truncation and completed seed-pair count. Include uncertainty over seed pairs;
small shifts alone are not proof of general competitive strength. Review defense
using the existing public replay/score diagnostics without changing the reward.

`trainer.evaluate()` still compares the original initialization with the current
learner. An imported 640-episode parent is not silently relabeled as the initial
model. Evaluate that retained parent explicitly for the curriculum A/B comparison.
The fresh-checkpoint evaluation tool labels saved-pool evaluation explicitly.
The existing replay exporter intentionally remains a uniform-opponent replay and
labels itself as such; it is a uniform diagnostic, not a replay of pool training.

## Verification

- `python -m unittest discover -s tests -p test_frozen_pool_training.py -v`
  covers exact partial NPZ resume, paired deterministic membership, independent
  RNG/state, public adapter separation, schemas 2–4, truncation, raw scores,
  read-only tensors, changed sources/contracts, bounded pools, malformed
  probabilities, explicit import provenance and manifest/resume rejection.
- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -q`
  is the full repository regression command.
- Baseline equivalence was also checked directly against e3be747: schemas 1–4,
  REINFORCE and PPO, five synthetic episodes including a partial batch. Policy,
  optimizer, RNG and NPZ checkpoint bytes were identical. All game records were
  identical excluding their necessarily variable elapsed wall time.

These checks establish infrastructure behavior on synthetic fixtures only; they
provide no learning results and do not measure real-scenario throughput.
