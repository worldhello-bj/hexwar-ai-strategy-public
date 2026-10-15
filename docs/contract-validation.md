# Offline contract validation

This change set fixes software correctness and isolation. It does not provide a trained model or establish competitive strength.

## Run

From the repository root, Python 3.11 or newer, without third-party packages:

```sh
python -m unittest discover -s tests -v
python tools/verify_rules.py
python tools/verify_tables.py
python -m env.match_eval --output /tmp/match-smoke.json --seeds 0 1 2 --max-seconds 60
```

The evaluator's default smoke scenario uses synthetic unarmed pieces and random legal actions. A truncated match is not a win or loss. It is an execution test, not an agent-strength evaluation. The evaluator uses alternating actions, not simultaneous submissions. Strategies receive JSON data; this is not an operating-system sandbox for untrusted Python code.

## Changes to integration contracts

- `Game.step_seconds` is a read-only query. Successful movement calls `Game.commit_step` once before updating position.
- Each Game owns its random generator. Reading legal actions must not consume it.
- Invalid environment actions or batches are atomic no-ops.
- Successful transactions preserve existing Game and Unit object identities.
- Fog memory records only previously observed opposing pieces; this is distinct from globally permitted score information.
- Explicit zero score values remain zero.

## Security compatibility

Credential submission in the two updated login/capture helpers requires HTTPS and rejects automatic redirects. The previous default HTTP login will stop; this change does not establish that the external site offers HTTPS. These helpers are not used by the offline tests. Cookies and query values are no longer included in the modified logs. Other HAR/export tools have not received comprehensive privacy hardening; their output may still contain sensitive data.

The local service remains bound to loopback. Do not expose it publicly without a separate authentication, request-limit and deployment review. Official client assets, account sessions and private captures are not part of this change.
