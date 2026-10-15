# Optional schema 4: shared public objective contacts

This isolated prototype extends schema 3 with three Boolean candidate fields
under `public_context.objective_contacts`:

- `intended_endpoint_is_public_objective`
- `intended_endpoint_is_unowned_public_objective`
- `acting_unit_is_ground_type1or2`

The intended endpoint is the last supplied move path cell, or the acting unit's
public position for other actions. A missing acting unit/position means no
endpoint. Unowned means an explicit public `held_by: null`; enemy ownership,
empty ownership strings and missing ownership do not qualify. Aggregation is
across public objectives and its Boolean keys contain no objective IDs. The
existing generic Boolean encoder therefore shares these identities across IDs.
The original schema-3 per-objective geometry remains alongside the flags.

These are observation relations, not tactics or execution predictions. Type 1
or 2 includes artillery and does not imply capture eligibility, sufficient
health, dismounted status, uncontested location, reachable endpoint, or a
successful capture. No engine eligibility query, hidden state, extra engine
simulation, reward shaping, action filtering, preferred unit, coordinates,
weights or script is added. Raw terminal rewards are unchanged.

## Opt-in and checkpoints

Use `--observation-schema 4` in the existing runner. The runner default remains
2; the general trainer default remains 1. Representation `public_v4_contacts`
is stored in the policy and trainer checkpoints. Replay export and evaluation
select the matching adapter, including mixed schema-3/schema-4 frozen matches.
Malformed/missing contact fields and schema/representation mismatches are
rejected. Start fresh models for schema 4: never relabel a schema-3 checkpoint
or import its trained weights as a matched fresh treatment.

## Controlled experiment plan (not launched)

Compare fresh schema 3 versus fresh schema 4 with the same complete map,
initialization seeds, training episode seed range, alternating seats, uniform
random opponent, episode and candidate-generation limits, model dimensions,
projection backend, optimizer/PPO configuration, public_own_score baseline,
raw-terminal scoring version and fixed completed-episode budget. Change only
schema/representation and isolated output locations. Do not warm-start from the
already-trained control, compare unequal training ages, or reuse held-out seeds
for training. A practical initial paired budget is 160 completed episodes per
arm, matching the prior diagnostic horizon, repeated across multiple model
seeds before claiming improvement. Budget and launch require a separate decision.

At fixed checkpoints, use identical fresh held-out seed pairs and both seats.
Report paired raw own score and score difference, W/D/L, completion/truncation
rates, and seed-level uncertainty. Compare each trained model with its own
fresh initialization as well as the paired other-schema model.

Diagnostic replay measures: total probability mass and selected actions with
intended objective endpoints, separately for unowned objectives and ground
acting types; actual movement displacement; actual ownership transitions;
policy entropy; candidate multiplicity and episode runtime. Keep intention,
selection, actual arrival and successful capture separate. Diagnostics never
feed rewards or action selection. If contact mass rises without captures,
inspect execution/contest/eligibility evidence rather than interpreting the
flag as a guarantee. Preserve all checkpoints and report incomplete episodes.

## Verification

The new tests cover endpoint/path semantics, explicit null ownership, shared
keys under objective-ID changes, type 1/2 including artillery/low-health/loaded
units, hidden enemy perturbation, detached views, no engine eligibility reads,
schema-3 relation preservation, exact delegated terminal rewards, schema 1–3
byte-identical features and checkpoint/RNG behavior against e1f85ee, runner
opt-in/defaults, checkpoint roundtrip/mismatch rejection, frozen replay
probability reconstruction and mixed-schema frozen evaluation.

Full command: `OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -q`.
Existing synthetic optimizer regression tests run as part of verification; no
new training experiment or active-tree change is part of this prototype.

Validation result: all 281 tests passed in 6.939 seconds; `git diff --check` passed.
