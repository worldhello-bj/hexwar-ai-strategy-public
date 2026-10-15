# Passive per-episode public behavior counters

Opt in with `--behavior-diagnostics` on `tools/cloud_train.py` or
`python -m env.neural_train`, or `trainer.train(n, behavior_diagnostics=True)`.
Default off. Independent of `--training-diagnostics` (optimizer statistics).
Enabled training records contain `behavior_diagnostics` schema version 1.
Cloud run manifests and standalone reports record `behavior_diagnostics_enabled`.
A cloud resume must keep the same flag; a new-run import can change it. Imported
historical rows remain unchanged and may lack counters. No historical actions
can be reconstructed from scores or previously recorded decision counts.

The counter receives only the learner observation already supplied to sampling
and its selected raw action. It has no episode, model, or live-engine reference.
It never fetches another observation, invokes another forward pass, consumes RNG,
changes reward, adds a game, or expands policy inputs. Nothing is checkpointed.
Evaluation and the common evaluation helper are unchanged. CPU reporting overhead
and elapsed timings may differ. Training behavior, tensors, optimizer, sampling,
and checkpoint contents must remain identical with the flag on or off.

## Field definitions

- `learner_action_type_counts`: selected learner actions, by exact action type.
  An `attack` count means an attack request, not a hit, damage, kill, or ongoing
  fire tick. Other fire action types remain separate; opponent actions excluded.
- `unique_move_requested_units`: distinct own unit IDs receiving a selected move
  request. This is not the number of units that actually moved.
- `requested_move_path_cell_count`: sum of supplied path list lengths for move
  requests, including starting cells when present and repeated cells.
- `requested_move_path_consecutive_segment_count`: sum of max(0, path length - 1).
  Current generated paths include the origin. This metric only counts adjacent
  entries in each supplied list. It does not infer a missing origin, validate
  adjacency, report hex distance, or assert successful execution/travel.
- `visible_contact_observation_count`: number of learner decision observations
  containing `enemy_visible`; the denominator for contact presence.
- `visible_contact_present_decisions`: those observations with at least one
  currently visible enemy. Stale/remembered `enemy_known` is excluded. This is a
  decision-sampled count, not contact duration or continuous battlefield coverage.
- `own_position_comparisons_between_learner_observations`: own-unit pairs with a
  public `pos` in both consecutive supplied learner observations, matched by ID.
- `own_position_changes_between_learner_observations`: number of such pairs with
  different positions; a unit may contribute multiple changes.
- `unique_own_units_with_observed_position_change`: distinct units among those
  changed pairs. May include transport or other causes, not attributed to moves.
- `own_hex_displacement_between_learner_observations_sum`: sum of odd-r geometric
  hex distances for those pairs. This is endpoint displacement, not actual path
  length, movement cost, or cumulative executed movement. Intermediate travel
  and returns to the same cell are invisible.
- `public_objective_owner_comparisons`: objective-ID pairs with `held_by` present
  in both consecutive supplied learner observations (null is a known unheld state).
- `public_objective_owner_changes_between_learner_observations`: changed owner
  values in those pairs. Includes either side's changes during the interval;
  cannot attribute a capture to an action or recover multiple intermediate flips.

Position/ownership observations are taken before learner decisions. There is no
extra initial, opponent-turn, or terminal observation. The first observation is a
baseline, missing/new/disappeared entities do not create a comparison, and changes
after the last learner observation are unobserved. Counts cover completed and
truncated attempts; terminal-at-reset attempts have zero decisions and counters.
When comparing runs, retain the denominators, seat, completion/truncation status,
and number of rows with this field. Do not treat absent legacy fields as zero.

## Validation

`python -m unittest tests.test_behavior_diagnostics -v` exercises exact on/off
checkpoint, outcomes, selected actions, policy input, public observation/step
trace, and forward-count parity for schemas 1–4, REINFORCE/PPO, NumPy/CSR. Small
synthetic fixtures also check field definitions, detached input non-mutation,
strict public-field access, truncation, flag default, manifest and resume guard.
Frozen-opponent tests also compare both flags together, compact pending-buffer
resume, immutable opponent tensors and unchanged source checkpoint bytes.
These are synthetic regression tests, not a new training experiment.
