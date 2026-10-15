"""Replay-shaped pending serialization stress test; no games or policy updates."""
import copy
import json
from pathlib import Path
import resource
import time
import zipfile
from env.neural_train import NeuralTrainer

root = Path('analysis/compact-benchmark')
replay_path = Path('/workspace/shared/hexwar-public-cloud/runs/cloud-training/publicscore-s3-seed7527/publicscore-s3-seed7527/replay.json')
trainer = NeuralTrainer.load(root / 'completed160.npz', 'scenario/7527', allow_real=True)
replay = json.loads(replay_path.read_text())
trajectory = []
for row in replay['decisions']:
    observation = dict(row['observation'])
    observation.pop('board_ref', None)
    observation['board'] = replay['public_board']
    candidates = [{k: candidate[k] for k in ('action', 'public_context')} for candidate in row['candidates']]
    trajectory.append(trainer.policy._decision(observation, candidates, row['selected_index']))
trainer.pending = [(copy.deepcopy(trajectory), replay['ending']['own_original_score']) for _ in range(7)]
trainer.episodes_attempted += 7
trainer.episodes_completed += 7
candidate_counts = [len(d.candidates) for d in trajectory]
feature_bytes = sum(x.nbytes for history, _ in trainer.pending for d in history for pair in (d.observation, *d.candidates) for x in pair)
del replay, trajectory, candidates, observation
start = time.perf_counter()
path = root / 'replay-pending7.npz'
trainer.save(path)
save_seconds = time.perf_counter() - start
save_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
with zipfile.ZipFile(path) as z:
    members = len(z.infolist())
    metadata_bytes = z.getinfo('metadata.npy').file_size
start = time.perf_counter()
resumed = NeuralTrainer.load(path, 'scenario/7527', allow_real=True)
load_seconds = time.perf_counter() - start
import numpy as np
assert trainer.policy.rng.bit_generator.state == resumed.policy.rng.bit_generator.state
assert trainer.episodes_completed == resumed.episodes_completed
for a, b in zip(trainer.pending, resumed.pending):
    assert a[1] == b[1]
    for da, db in zip(a[0], b[0]):
        assert da.selected == db.selected and da.public_score_offset == db.public_score_offset
        for pa, pb in zip((da.observation, *da.candidates), (db.observation, *db.candidates)):
            for xa, xb in zip(pa, pb): np.testing.assert_array_equal(xa, xb)
for label in ('params', 'm', 'v'):
    for key, value in getattr(trainer.policy, label).items():
        np.testing.assert_array_equal(value, getattr(resumed.policy, label)[key])
print(json.dumps(dict(trajectories=7, decisions_per_trajectory=len(candidate_counts),
    candidate_counts=candidate_counts, feature_bytes=feature_bytes, archive_members=members,
    metadata_bytes=metadata_bytes, checkpoint_bytes=path.stat().st_size, save_seconds=save_seconds,
    save_process_peak_rss_bytes=save_peak, load_seconds_with_source_still_resident=load_seconds,
    exact_pending_and_policy=True), sort_keys=True))
