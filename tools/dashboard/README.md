# Offline training dashboard

Dependency-free Python 3.10+ read-only artifact viewer. It never starts training, imports model weights, contacts game servers, or hosts publicly.

Run from the repository root:

    python tools/dashboard/server.py --port 8765

Open http://127.0.0.1:8765 in the browser on the same cloud machine. A cloud loopback URL is not automatically accessible from the user's computer. Do not imply otherwise or expose it publicly without a separately authorized access setup.

Custom bounded artifact directory:

    python tools/dashboard/server.py --runs runs/cloud-training --port 8765

The default root is `runs/cloud-training`. Scanning is limited to three directory levels, 500 JSON artifacts, and 2,000 directory entries per poll; a visible warning indicates limits. Files containing `checkpoint` or `model` in their name, symlinks and unrecognized JSON schemas are ignored. Each JSON file has a 32 MiB read limit. Cached summaries are refreshed on mtime/size changes. Completed reports are visible only after the writer produces them; atomic file replacement is recommended. Polling is every five seconds. The server is hard-bound to IPv4 loopback and rejects other Host headers, mutation methods, and arbitrary file requests.

## Supported actual evidence

- `progress.json`: status, updated_at, commit, run_id, phase, episodes_attempted, episodes_completed, episodes_target, optimizer_updates (or updates), elapsed_seconds, episodes_per_minute, resources, latest_report, latest_evaluation, issues. Missing values remain unavailable. The budget progress bar uses attempted episodes when available. A stale progress artifact does not prove the trainer is alive.
- Native `env.neural_train` reports: episode/update totals, original terminal score summaries, heldout initial/trained policies versus random, resource measurements and rollout throughput calculated from recorded rollout durations. Current RSS and peak RSS remain distinct fields.
- Fresh frozen evaluations: paired trained-minus-initial original-score deltas, complete seed pairs and descriptive standard errors, without claiming statistical significance.
- Frozen head-to-head reports: A/B W/D/L and raw scores, with full source caveats.
- Replay exports: selected action, original observed state, selected probability, public score review points and terminal/truncation state. Candidate lists are omitted from the review payload. Position changes are measured between consecutive learner observations for shared own-unit IDs; they are not commanded-move counts or path length. Error rate is unavailable unless recorded explicitly under `error_metrics`; it is never inferred from losing or standing still.

Report times use UTC+8. Training means are plotted only within the same directory and scoring mode/version; even these are descriptive batch scores, not comparable strength benchmarks. The dashboard does not invent a commit: it shows the runner's recorded commit under current metadata.

## Validation

    python -m unittest discover -s tools/dashboard -p 'test_*.py' -v
    node --check tools/dashboard/app.js

All unit-test artifacts are generated in a temporary directory and explicitly constitute fixtures, not training evidence. Tests cover empty-state honesty, report compaction, exact rollout throughput, malformed/partial files, symlink exclusion, observed movement counts, replay traversal protection, loopback Host validation, read-only HTTP and static page delivery.
