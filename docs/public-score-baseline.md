# Optional current public own-score baseline

Fresh runs may use `--baseline-offset public_own_score` (or the same
`baseline_offset` in the run configuration). This is mutually exclusive with
`prior_batch_mean`. No old checkpoint is converted; resume must inherit the
saved mode. Both old modes retain their serialized bytes and arithmetic.

## Objective and source boundary

For decision t, let G be the original terminal own-seat score, c_t be the
CURRENT pre-action public observation's `score.own`, and v_theta(o_t) be the
observation-only residual critic. Regress v onto G - c_t and use detached
advantage G - c_t - v_old(o_t). Thus the complete value estimate is
c_t + v_theta(o_t). No intermediate reward, action bonus, preset strategy,
discount, reward scaling, or terminal-score modification is introduced.

For an action-independent baseline b(o), conditional on the current observation
and fixed pre-update parameters,

    E_a[b(o) grad log pi(a|o)] = b(o) grad sum_a pi(a|o) = 0.

Subtracting this baseline therefore leaves the expected REINFORCE policy
gradient of raw terminal return unchanged. Past actions can affect the current
score; that does not make it depend on the action currently being sampled.
Advantages are detached: this identity does not authorize differentiating
through the critic or score. The shared encoder still receives its existing
auxiliary critic-regression gradient. PPO uses the same frozen advantages for
all epochs, with its usual clipped surrogate: PPO clipping is not itself an
unbiased policy-gradient estimator, and baseline changes can change its finite
batch update. This feature is a variance-control hypothesis, not a strength claim.

The ONLY source is the observation supplied to `_decision`, specifically
`observation['score']['own']`. `TwoPlayerEpisode.observe` explicitly exposes a
totals-only public scoreboard when `public_score=True`; schema 2 and 3 adapters
copy that field. No private score decomposition, opposing observation, hidden
engine state, or post-action/terminal information is fetched to form c_t.
This offline public-score contract does not assert that an online UI exposes it.
Missing, boolean, nonnumeric, or nonfinite public scores fail explicitly.

## Persistence and compatibility

Each sampled Decision stores `public_score_offset`, including an explicit zero.
Old decisions omit that field and load as None. Their reserialization is
unchanged. Public-score mode rejects missing offsets; old modes reject populated
offsets. Pending trajectories preserve these offsets on JSON resume. The public
mode has no running-return baseline state. The trainer rejects public_score=False
up front, and CLI resume rejects a mode change. Raw returns in pending batches,
rollout reports, terminal rewards, and evaluation summaries are unchanged.

## Verification and experiment plan

`python -m unittest discover -s tests -q`: 274 tests pass, including new coverage:
all-parameter finite differences for REINFORCE and both signs/all PPO clipping
branches; frozen offsets/advantages across PPO epochs; unchanged raw returns;
missing-score and malformed checkpoint validation; exact interrupted-batch
resume; actual schema 1/2/3 public observation collection. Existing independent
pre-PPO fixture checks retain bit-identical old-mode outputs and checkpoints.

Constant-offset test: hold encoded features fixed, add a per-trajectory constant
to BOTH G and every c_t, and get bit-identical loss/update/checkpoint. Changing
the observation and re-encoding its score changes input features, so whole-model
invariance to arbitrary scoreboard relabeling is deliberately not claimed.

Next experiment should start two fresh matched policies per seed, with identical
initial parameters, architecture, rollout/opponent seeds, episode budget, PPO
settings, scoring mode, projection backend, and heldout seeds; vary only
prior_batch_mean versus public_own_score. Evaluate both seats, aggregate paired
seed deltas in raw terminal own-score, and inspect critic targets/advantages by
seat. Do not reuse a historical-mean critic as a residual public-score critic.
This prototype runs no mass training and makes no old-checkpoint conversion.
