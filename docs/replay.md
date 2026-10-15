# Frozen learned-policy replays

`tools/export_policy_replay.py` exports a **sampled trajectory** of a saved neural
policy in the original synthetic offline board game. It does not enumerate the
neural network's complete strategy or introduce handwritten tactics. No training,
weight updates, reward shaping, or checkpoint rewriting occurs.

```sh
OPENBLAS_NUM_THREADS=1 python tools/export_policy_replay.py \
  --checkpoint /path/to/checkpoint-100.json \
  --snapshot samples/demo --seed 2700000 --learner-side BLUE \
  --report /path/to/replay-2700000-blue.json \
  --csv /path/to/replay-2700000-blue.csv \
  --markdown /path/to/replay-2700000-blue.md
```

Repeat with `--learner-side RED` and new output names for the other seat. All
outputs must be new files. The optional Markdown is a concise selected-choice
table plus review points; JSON and CSV retain all candidate probabilities. Repeating the same checkpoint, snapshot, seed and seat
reproduces the JSON (there are no wall-clock timing fields). Sampling follows
`NeuralTrainer._rollout(training=False)` exactly, including separate derived RNG
seeds for the learned policy and uniform-random opponent. No seeds or outcomes
are selected automatically. A chosen seed may overlap training; the caller must
choose a fresh seed when interpreting this as an unseen-game illustration.

## Reading the outputs

- JSON contains each actual learner decision: original game time, complete
  permitted seat observation, every legal candidate and its probability, the
  selected candidate index, and the unchanged original action sent to the game.
  Schema 2 also preserves each candidate's public spatial context. Its static
  board is stored once as `public_board`; each observation has
  `board_ref: "#/public_board"`. The `reconstruct_observation` helper restores
  the exact full policy input; dynamic observations remain in every decision.
- CSV has one row per legal candidate, grouped by decision index. `selected=True`
  identifies the action actually taken. Original actions are JSON strings so
  paths, IDs and parameters retain their exact values. The complete observations
  remain in JSON; match them by decision index.
- `review_points` flags every public aggregate score change after either seat's
  step, plus game end or truncation. A score-change marker's
  `latest_learner_decision_index` is only a navigation reference, **not causal
  attribution**. It may be null if the learner has not acted yet. These markers
  are review aids, not evidence that a decision was optimal or decisive.
- `ending.own_original_score` is the unmodified original terminal score, or null
  if truncated. `public_score` is included only when the game exposes it. A
  truncation's public scoreboard is an interim observation, never a final payoff.
- Public-score-disabled checkpoints do not export opponent totals, terminal
  opponent payouts, winner, score differences or private score components.

Only the learner's normal observation API is serialized. Hidden opponent units,
private opponent state and opponent actions are not exported. Visible units and
legitimate stale last-seen records remain visible as prescribed by the game.
The opponent still receives its own legal candidates to play uniformly at random;
those candidates are not written into the replay. The Python API boundary is not
an adversarial security sandbox.

The exporter checks the entire loaded trainer checkpoint (including parameters,
optimizer state, RNG and counters), the source checkpoint bytes, and snapshot
manifest after the run. Both supported observation schemas are tested. Focused
checks also compare all executed actions against the existing evaluation runner,
perturb hidden enemy state, verify score visibility and verify CSV fidelity:

```sh
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -p test_policy_replay.py -v
```

One replay, including a win or loss, is not a strength estimate. Paired multi-seed
held-out evaluation remains the appropriate separate measurement.
