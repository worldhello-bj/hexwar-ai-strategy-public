# Frozen neural head-to-head evaluation

This is evaluation only on original synthetic game data. It adds no tactical
features, reward shaping, real-world calibration, optimizer updates or training
opponent changes. Existing trainer and environment files are unchanged.

Run from the repository root with the optional neural dependencies installed:

```sh
OPENBLAS_NUM_THREADS=1 python tools/evaluate_frozen_opponents.py \
  --checkpoint-a /path/to/checkpoint-500.json \
  --checkpoint-b /path/to/checkpoint-100.json \
  --snapshot samples/demo --seed-start 2900000 --seed-count 64 \
  --report /path/to/new-head-to-head-report.json
```

Use `--b-initial` to compare A against B's saved initial policy instead. Both
checkpoints must match the supplied synthetic snapshot and the same episode
configuration, including public-score visibility, game clock and safety budget.
Each can use observation schema 1 or 2. Seeds must be fresh with respect to both
training-attempt intervals and reserved evaluation sets. Reports are never
overwritten. The caller must also avoid seeds from earlier external evaluations;
checkpoint metadata cannot record those separate reports.

Every scenario seed runs twice, exchanging policy seats. BLUE still moves first.
Each seat has its own independently seeded NumPy sampler, derived from scenario
seed, seat and policy label. It samples only the model probabilities over supplied
legal candidates. Model training RNG is never sampled. Both policies receive only
their own fog-filtered detached observations; schema 2 uses the same public-board
and candidate adapter as training. Original terminal scores are unchanged.

> The evaluator never alters payoffs, but the payoffs themselves changed on
> 2026-10-15: `remaining` now follows the back-derived platform formula
> (`Σ score × blood` for deployed units plus carried units at establishment
> strength), so reports produced before that commit are not comparable.
> See `docs/计分规则待确认.md`.

Reports include source hashes, frozen model-state hashes, snapshot hashes,
configuration, raw scores by physical seat, own score for A and B, score difference,
win/draw/loss, terminal status and public input/action trace hash. State hashes
cover parameters, optimizer state, baseline state and training RNG. These are
verified after every game; source files are verified again after the evaluation.
The transient public-board feature cache is not serialized learned state.

W/D/L is determined by raw terminal score comparison, matching neural evaluation;
the environment's natural terminal winner is separately preserved in status.
Safety-budget truncations yield no score, draw or win. Paired summary statistics
exclude the whole seed pair if either seat truncates. The standard error treats
seed pairs, not individual games, as units. Fixed seeds and small samples remain
descriptive; no confidence interval or claim of improved strength is implied.
Same-policy matches have symmetric expectations after seat balancing, but
independent sampling and finite samples need not produce exactly equal wins.

Tests:

```sh
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -p test_frozen_opponents.py
```
