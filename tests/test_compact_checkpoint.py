"""Compact transport parity and hostile-input regression tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from env.compact_checkpoint import read_checkpoint, save_compact
from env.match_eval import write_synthetic_snapshot
from env.neural_policy import NeuralPolicy
from env.neural_train import NeuralTrainer


class CompactTests(unittest.TestCase):
    def test_exact_resume_partial_batches_schemas_and_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = write_synthetic_snapshot(root / 'snapshot')
            for schema in (2, 3, 4):
                for baseline in ('none', 'prior_batch_mean', 'public_own_score'):
                    for mode in ('reinforce', 'ppo_clip'):
                        with self.subTest(schema=schema, baseline=baseline, mode=mode):
                            def trainer():
                                return NeuralTrainer(snapshot, observation_schema=schema, batch_episodes=2,
                                    policy_config=dict(buckets=8, hidden_dim=2, baseline_offset=baseline, update_mode=mode))
                            def rollout(t, policy, seed, side, *, training):
                                _, decision = policy.sample({'schema_version': schema, 'board': {}, 's': seed,
                                    'score': {'own': seed + .25}},
                                    [{'action': i, 'public_context': {'objective_relations': [], **({'objective_contacts': {
                                        'intended_endpoint_is_public_objective': False,
                                        'intended_endpoint_is_unowned_public_objective': False,
                                        'acting_unit_is_ground_type1or2': False}} if schema == 4 else {})}} for i in (1, 2)])
                                return {'own_score': None if seed in (2, 5) else seed * 10}, [decision]
                            with patch.object(NeuralTrainer, '_rollout', rollout):
                                whole = trainer(); whole.train(8)
                                partial = trainer(); partial.train(4)
                                self.assertEqual(len(partial.pending), 1)
                                name = f'{schema}-{baseline}-{mode}'
                                j, n = root / (name + '.json'), root / (name + '.npz')
                                partial.save(j); partial.save(n)
                                resumed = NeuralTrainer.load(n, snapshot)
                                legacy = NeuralTrainer.load(j, snapshot)
                                self.assertEqual(resumed.checkpoint(), legacy.checkpoint())
                                self.assertEqual(resumed.checkpoint(), partial.checkpoint())
                                roundtrip = root / (name + ".roundtrip.json")
                                resumed.save(roundtrip)
                                self.assertEqual(roundtrip.read_bytes(), j.read_bytes())
                                resumed.train(4)
                                self.assertEqual(resumed.checkpoint(), whole.checkpoint())
                                with self.assertRaises(FileExistsError):
                                    partial.save(n)

    def test_policy_exact_and_no_float_lists_on_compact_save(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.npz'
            policy = NeuralPolicy(buckets=8, hidden_dim=2)
            expected = policy.checkpoint()
            policy.save(path)
            self.assertEqual(NeuralPolicy.load(path).checkpoint(), expected)
            self.assertIsInstance(read_checkpoint(path)['params']['observation'], np.ndarray)

    def test_corruption_and_semantic_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = write_synthetic_snapshot(root / 'snapshot')
            trainer = NeuralTrainer(snapshot, policy_config=dict(buckets=8, hidden_dim=2))
            original = trainer.checkpoint(arrays=True)
            mutations = [lambda d: d.update(scoring_version='wrong'),
                         lambda d: d['snapshot'].update(sha256='wrong'),
                         lambda d: d.update(episodes_completed=1),
                         lambda d: d['policy'].update(steps=1),
                         lambda d: d['policy']['params'].update(value=np.zeros(3)),
                         lambda d: d['policy']['params']['value'].__setitem__(0, np.nan)]
            for i, mutate in enumerate(mutations):
                data = copy.deepcopy(original); mutate(data)
                path = root / f'bad{i}.npz'; save_compact(path, data)
                with self.assertRaises(ValueError):
                    NeuralTrainer.load(path, snapshot)
            good = root / 'good.npz'; trainer.save(good)
            with np.load(good, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            for i, mutate in enumerate([
                    lambda a: a.update(a000000=a['a000000'].astype(np.float32)),
                    lambda a: a.update(a000000=np.array([object()], dtype=object)),
                    lambda a: a.update(metadata=np.array([1.], dtype=np.float64)),
                    lambda a: a.update(unexpected=np.zeros(1)),
                    lambda a: a.update(metadata=np.frombuffer(b'{"format":"wrong"}', dtype=np.uint8)),
                    ]):
                altered = dict(arrays); mutate(altered)
                path = root / f'archive{i}.npz'; np.savez(path, **altered)
                with self.assertRaises((ValueError, KeyError)):
                    NeuralTrainer.load(path, snapshot)

    def test_metadata_refs_and_declared_allocation_limits(self):
        import io
        import zipfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / 'good.npz'
            NeuralPolicy(buckets=8, hidden_dim=2).save(good)
            with np.load(good, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            envelope = json.loads(arrays['metadata'].tobytes())
            for i, mutate in enumerate([
                    lambda e: e.update(version=True),
                    lambda e: e['data']['params'].update(value={'$array': 'missing'}),
                    lambda e: e['data']['params'].update(value={'$array': 'a000000'}),
                    lambda e: e['data']['config'].update(buckets=10**12),
                    ]):
                e = copy.deepcopy(envelope); mutate(e)
                altered = dict(arrays)
                altered['metadata'] = np.frombuffer(json.dumps(e).encode(), dtype=np.uint8)
                path = root / f'ref{i}.npz'; np.savez(path, **altered)
                with self.assertRaises(ValueError):
                    NeuralPolicy.load(path)
            # Reject huge declared allocations before numpy can allocate them.
            header = io.BytesIO()
            np.lib.format.write_array_header_1_0(header,
                dict(descr='<f8', fortran_order=False, shape=(10**12,)))
            path = root / 'oversize.npz'
            with zipfile.ZipFile(good) as source, zipfile.ZipFile(path, 'w') as target:
                for name in source.namelist():
                    target.writestr(name, header.getvalue() if name == 'a000000.npy' else source.read(name))
            with self.assertRaises(ValueError):
                NeuralPolicy.load(path)

    def test_packed_pending_offsets_and_member_bound(self):
        import zipfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = write_synthetic_snapshot(root / 'snapshot')
            trainer = NeuralTrainer(snapshot, policy_config=dict(buckets=8, hidden_dim=2))
            decision = trainer.policy._decision({}, list(range(600)))
            trainer.pending = [([decision], 3)]
            trainer.episodes_attempted = trainer.episodes_completed = 1
            path = root / 'packed.npz'; trainer.save(path)
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(len(archive.infolist()), 46)  # 42 policy + 3 sparse + metadata
            self.assertEqual(NeuralTrainer.load(path, snapshot).checkpoint(), trainer.checkpoint())
            with np.load(path, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            metadata = json.loads(arrays['metadata'].tobytes())
            key = metadata['data']['pending'][0]['decisions'][0]['$packed_features']['offsets']['$array']
            for i, mutation in enumerate((lambda a: a.__setitem__(0, 1),
                                           lambda a: a.__setitem__(-1, a[-1]+1),
                                           lambda a: a.__setitem__(2, -1))):
                bad = dict(arrays); bad[key] = arrays[key].copy(); mutation(bad[key])
                target = root / f'bad-offset{i}.npz'; np.savez(target, **bad)
                with self.assertRaises(ValueError):
                    NeuralTrainer.load(target, snapshot)

    def test_state_digest_exact_structure_and_tensor_mutations(self):
        from env.compact_checkpoint import state_digest
        policy = NeuralPolicy(buckets=8, hidden_dim=2)
        original = policy.checkpoint(arrays=True)
        digest = state_digest(original)
        self.assertEqual(digest, state_digest(copy.deepcopy(original)))
        for mutate in (lambda d: d['params']['value'].__setitem__(0, 1),
                       lambda d: d['m']['value'].__setitem__(0, -0.),
                       lambda d: d.update(steps=1),
                       lambda d: d['rng'].update(has_uint32=1)):
            changed = copy.deepcopy(original); mutate(changed)
            self.assertNotEqual(state_digest(changed), digest)
        self.assertNotEqual(state_digest([1, 23]), state_digest([12, 3]))
        self.assertNotEqual(state_digest({'a': [1]}), state_digest({'a': 1}))
