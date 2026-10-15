# Optional public spatial relations (schema 3)

This isolated prototype starts from commit `38fc987`. It does not change the
active baseline, engine, full map, action enumeration, legal action ordering,
reward, scoring mode, terminal return, optimizer, or opponent.

## Input contract

Opt in with `--observation-schema 3`. Schema 1/2 remain unchanged; schema 2 is
still `PublicObservationAdapter`. Schema 3 uses `SpatialObservationAdapter` and
checkpoint representation `public_v3_spatial`.

For each action with an observed own `unit`, candidate public context adds
`objective_relations`, keyed by the existing public objective ID, containing:

- `origin_distance`: numeric hex distance from the observed acting unit
- `intended_destination_distance`: numeric distance from the last move path cell
  (or the origin for non-move actions)
- `distance_change`: destination minus origin distance; negative means closer
- `held_by`: the objective's already-public ownership, unchanged

Actions without an own acting unit, including wait and group area fire, have
an empty relation dictionary. This is intentional: the prototype does not
invent a group origin or designate a preferred unit. All public objectives are
included, without ranking, preferred coordinates, action bonuses, or scripted
choices. Distance is purely geometric and does not imply reachability,
successful displacement, capture eligibility, or a reward estimate. In
particular, blocked movement may not reach the intended endpoint. Movement can
auto-capture cities, so an absent explicit capture candidate is not evidence of
an absent scoring opportunity.

The function uses only the supplied action and existing public observation;
it never reads enemy authoritative state or performs a hypothetical engine
step. The underlying schema-2 static board whitelist is reused unchanged.

## Policy/checkpoint compatibility

The neural encoder remains generic signed-log JSON feature hashing with two
independently normalized banks. Schema 3 uses the same tensor dimensions and
parameter initialization as schema 2 but has a distinct representation marker.
Schema-specific policy checks and trainer resume checks reject mismatches.
Do not relabel a schema-2 checkpoint as schema 3: hashed feature semantics have
changed even though dimensions match. Start a fresh paired ablation. Existing
schema-2 checkpoint configuration and feature vectors remain bit-identical.

## Controlled next ablation (not launched)

Compare fresh schema-2 and schema-3 policies on the identical full real map,
using the same seed pairs, alternating seats, uniform-random opponent, complete
episode budget, raw terminal scoring mode, model width, PPO settings, baseline
offset, held-out seeds and action candidate list. Use the current baseline's
configuration as the control; change only observation schema and output paths.
Do not resume the already trained baseline into schema 3 or compare unequal
training ages as if paired. Budget and run launch remain the parent's decision.

Measure held-out paired own score, score difference, W/D/L and truncations,
separately by seat, against both the matched fresh initialization and the
trained control. Record actual movement displacement and actual ownership
transitions. For diagnostic replay, measure probability mass on moves ending
at each public objective and actual selections (not just explicit capture
counts); this is an evaluation diagnostic, never a reward term. Report movement
entropy/candidate multiplicity too. Repeat training seeds before attributing a
strength improvement to geometry. Stop each planned run at its fixed completed
episode budget, preserving checkpoints and reporting incomplete episodes.

## Verification

`tests/test_spatial_observation.py` checks even/odd geometry, sign conventions,
nonmovement semantics, hidden-state perturbation invariance, detachment,
unchanged actions and public board, identical schema-2 sparse feature vectors
and probabilities against source from `38fc987`, exact delegated rewards and
terminal scores, checkpoint roundtrip, and explicit schema mismatch rejection.
No experiment training run has been launched by this prototype work.

Validation result: full offline suite passes, 238 tests in 5.575 seconds with
`OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -q`. This includes
small synthetic optimizer regression tests from the existing suite, not a new
training experiment. Six new schema-3 tests include frozen replay probability
reconstruction and mixed-schema frozen-opponent execution. Replay and frozen
opponent tools now choose the correct adapter for each checkpoint schema.
The unsupported-schema test was updated from 3 to 4 because 3 is now supported.
