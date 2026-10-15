# Passive batch diagnostics

Opt in with `--training-diagnostics` on `tools/cloud_train.py` or
`env/neural_train.py`; default is off. It is a runtime reporting choice, not a
model hyperparameter, and adds no checkpoint metadata or model features.
Python callers can use `trainer.train(n, diagnostics=True)` or
`policy.update_batch(trajectories, diagnostic_seats=[...])`.

Completed batch update records include `diagnostics.epochs`. Each epoch contains
ALL, BLUE, RED and UNKNOWN groups. Each group reports:

- Advantage count, nonzero count, positive/negative counts, mean, population
  standard deviation, min and max. PPO advantages are the exact frozen raw
  terminal return minus configured baseline and pre-update critic prediction;
  they repeat unchanged across epochs. The first epoch is the per-batch summary.
- Mean and distribution of categorical entropy from the already computed
  candidate probabilities/log probabilities at that epoch's parameters.
- Sampled ratio clip fraction: proportion of selected-action ratios outside
  [1 - epsilon, 1 + epsilon], compared in log space. This is distinct from
  `surrogate_clip_fraction`, which additionally depends on advantage sign.
- `sampled_approx_kl`: distribution of log pi_old(selected) minus
  log pi_current(selected), the sampled k1 estimate. A finite sample can be
  negative. This is not exact full-distribution KL, nor a convergence guarantee.

PPO epoch 1 is measured before the first optimizer update: ratio clip fraction
and sampled KL are zero. Subsequent epochs show movement from the frozen behavior
policy. The final epoch is also measured before that epoch's optimizer update;
there is deliberately no post-update forward pass. REINFORCE has no old-policy
ratio: its ratio/KL fields are null/empty. Empty groups have zero counts and null
means/fractions. Zero-advantage samples may be outside the ratio clip band while
having zero surrogate clipping and zero actor contribution.

Seat labels are ephemeral provenance supplied by the trainer, never extracted
from hidden game state or hashed features. Pending trajectories loaded from a
checkpoint without this provenance are UNKNOWN; newly collected ones have the
actual learner seat. Labels are tracked across train calls and cleaned up with
successful batches or PPO rollback. No attempt is made to reconstruct missing
seat labels from alternating episode counts when truncation might invalidate it.

## Invariants and validation

Diagnostics read existing forward tensors and accumulate scalar statistics.
They do not change loss/gradient arithmetic, parameters, optimizer moments,
RNG, rollout count, forward count, action candidates, reward, or checkpoint
format. They add CPU reporting overhead and small scalar-summary storage only.

Tests compare exact full checkpoints and update results with diagnostics on/off
for REINFORCE and PPO, NumPy and CSR, and none/prior_batch_mean/public_own_score
baselines. They verify identical forward and rollout counts, split pending
batches, compact checkpoint resume with UNKNOWN attribution, empty batches,
known ratio/KL/clip examples, and CLI opt-in defaults.

No training experiment was launched. Existing synthetic regression tests do
exercise optimizer updates as usual.

Standalone diagnostics validation: 293 tests passed in 6.586 seconds.
