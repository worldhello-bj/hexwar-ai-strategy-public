"""Small synthetic fixtures only; no real scenario experiment is launched."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from env.compact_checkpoint import state_digest, save_compact
from env.frozen_pool import describe_pool, FrozenPool, file_sha256
from env.match_eval import write_synthetic_snapshot
from env.neural_policy import NeuralPolicy
from env.neural_train import NeuralTrainer, derived_seed, summary
from env.public_observation import PublicObservationAdapter
from tools import cloud_train


def normalized(records):
    return [{k: v for k, v in r.items() if k != 'elapsed_seconds'} for r in records]


class FrozenPoolTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.snapshot = write_synthetic_snapshot(self.root / 'snapshot')
        self.config = dict(seed=17, batch_episodes=2, heldout_seeds=[100],
            observation_schema=2, episode_config={'max_seconds':20, 'max_steps':100},
            policy_config={'buckets':16, 'hidden_dim':3, 'update_mode':'ppo_clip'})
        self.source = NeuralTrainer(self.snapshot, **self.config)
        self.source_rows = self.source.train(2)
        self.path = self.root / 'opponent.npz'
        self.source.save(self.path)
        self.pool = describe_pool([self.path], include_uniform=True)

    def trainer(self, pool=None):
        return NeuralTrainer(self.snapshot, **self.config, opponent_pool=self.pool if pool is None else pool)

    def test_default_unchanged_and_pool_does_not_initialize_learner_rng(self):
        a = NeuralTrainer(self.snapshot, **self.config)
        b = NeuralTrainer(self.snapshot, **self.config, opponent_pool=None)
        c = self.trainer()
        self.assertEqual(state_digest(a.policy.checkpoint(arrays=True)), state_digest(c.policy.checkpoint(arrays=True)))
        self.assertEqual(normalized(a.train(3)), normalized(b.train(3)))
        self.assertEqual(state_digest(a.checkpoint(arrays=True)), state_digest(b.checkpoint(arrays=True)))
        self.assertNotIn('opponent_pool', a.checkpoint())
        self.assertNotIn('opponent_pool_import', a.checkpoint())

    def test_partial_npz_resume_exact_with_frozen_pool_and_raw_scores(self):
        full, split = self.trainer(), self.trainer()
        source_before = self.path.read_bytes()
        frozen_before = split._opponent_pool.state_digest()
        records = full.train(5)
        rows = split.train(3)
        self.assertEqual(len(split.pending), 1)
        checkpoint = self.root / 'partial.npz'; split.save(checkpoint)
        split = NeuralTrainer.load(checkpoint, self.snapshot)
        rows += split.train(2)
        self.assertEqual(normalized(records), normalized(rows))
        self.assertEqual(state_digest(full.checkpoint(arrays=True)), state_digest(split.checkpoint(arrays=True)))
        self.assertEqual(split._opponent_pool.state_digest(), frozen_before)
        self.assertEqual(source_before, self.path.read_bytes())
        self.assertEqual([r['learner_side'] for r in records], ['BLUE','RED','BLUE','RED','BLUE'])
        for row in records:
            self.assertEqual(row['own_score'], row['status']['final_scores'][row['learner_side']])
            self.assertNotEqual(row['learner_side'], row['opponent_side'])
            self.assertEqual(row['opponent_id'], self.pool['entries'][row['opponent_index']].get('sha256','uniform'))

    def test_evaluation_frozen_and_paired_selection(self):
        t = self.trainer()
        t.train(3)
        before = state_digest(t.checkpoint(arrays=True))
        frozen = t._opponent_pool.state_digest()
        with patch.object(NeuralPolicy, 'sample', side_effect=AssertionError('training RNG used')), \
             patch.object(NeuralPolicy, 'update_batch', side_effect=AssertionError('optimizer used')):
            result = t.evaluate()
        self.assertEqual(before, state_digest(t.checkpoint(arrays=True)))
        self.assertEqual(frozen, t._opponent_pool.state_digest())
        for label in ('initial','trained'):
            rows = result[label]['games']
            self.assertEqual(rows[0]['opponent_id'], rows[1]['opponent_id'])
        pool = t._opponent_pool
        self.assertEqual({pool.select(derived_seed(17,i,'pool-selection-v1')) for i in range(50)}, {0,1})
        for p in pool._policies:
            if p:
                for value in p.params.values():
                    with self.assertRaises(ValueError): value.flat[0] = 123

    def test_public_adapters_are_separate_and_own_seat_only(self):
        t = self.trainer(describe_pool([self.path]))
        calls = []
        original = PublicObservationAdapter.observe
        def observe(adapter, side):
            calls.append((adapter, side))
            return original(adapter, side)
        with patch.object(PublicObservationAdapter, 'observe', observe):
            t.train(2)
        # One adapter per seat within each episode. No adapter observes both seats.
        self.assertTrue(calls)
        by_adapter = {}
        for identity, side in calls: by_adapter.setdefault(identity,set()).add(side)
        self.assertTrue(all(len(sides)==1 for sides in by_adapter.values()))

    def test_source_and_contract_mismatch_rejected(self):
        t = self.trainer(); path = self.root / 'resume.npz'; t.save(path)
        original = self.path.read_bytes()
        self.path.write_bytes(original + b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'): NeuralTrainer.load(path, self.snapshot)
        with self.assertRaisesRegex(ValueError, 'changed'): t.train(1)
        self.path.write_bytes(original)
        for field, value in [('observation_schema',3),('scoring_version',-1),
                             ('scoring_mode','platform'),('snapshot',{})]:
            data = self.source.checkpoint(arrays=True); data[field] = value
            other = self.root / (field + '.npz'); save_compact(other,data)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError,'contract'):
                self.trainer(describe_pool([other]))
        data = self.source.checkpoint(arrays=True)
        data['policy']['config']['representation']='legacy_json'
        # Schema mismatch is rejected even if a valid alternative policy is supplied.
        data['policy'] = NeuralPolicy(buckets=16,hidden_dim=3).checkpoint(arrays=True)
        bad = self.root/'representation.npz'; save_compact(bad,data)
        with self.assertRaises(ValueError): self.trainer(describe_pool([bad]))

    def test_all_public_schemas_and_truncation(self):
        for schema in (2,3,4):
            config=copy.deepcopy(self.config); config['observation_schema']=schema
            source=NeuralTrainer(self.snapshot,**config);source.train(2)
            path=self.root/('schema-%s.npz'%schema);source.save(path)
            trainer=NeuralTrainer(self.snapshot,**config,opponent_pool=describe_pool([path]))
            before=trainer._opponent_pool.state_digest()
            self.assertEqual(len(trainer.train(2)),2)
            trainer.evaluate()
            self.assertEqual(before,trainer._opponent_pool.state_digest())
        config=copy.deepcopy(self.config);config['episode_config']['max_steps']=0
        data=self.source.checkpoint(arrays=True);data['episode_config']['max_steps']=0
        path=self.root/'truncated.npz';save_compact(path,data)
        trainer=NeuralTrainer(self.snapshot,**config,opponent_pool=describe_pool([path]))
        row=trainer.train(1)[0]
        self.assertIsNone(row['own_score']);self.assertIsNone(row['result'])
        self.assertEqual(trainer.pending,[]);self.assertEqual(trainer.updates,0)

    def test_fresh_evaluation_labels_saved_pool(self):
        from tools.evaluate_neural_checkpoint import evaluate_checkpoint
        trainer=self.trainer();trainer.train(2)
        path=self.root/'evaluated.npz';trainer.save(path)
        result=evaluate_checkpoint(path,self.snapshot,seed_start=200,seed_count=1)
        self.assertEqual(result['evaluation_type'],'fresh_frozen_neural_vs_fixed_pool')

    def test_checkpoint_pool_metadata_tamper_rejected(self):
        trainer=self.trainer()
        data=trainer.checkpoint(arrays=True)
        data['opponent_pool']['opponents'][0]['policy_config']['hidden_dim']=999
        path=self.root/'tampered.npz';save_compact(path,data)
        with self.assertRaisesRegex(ValueError,'pool differs'):
            NeuralTrainer.load(path,self.snapshot)

    def test_limits_and_invalid_probabilities(self):
        for paths, include in [([],False),([self.path]*2,False),([self.path]*8,True)]:
            with self.assertRaises(ValueError): describe_pool(paths,include_uniform=include)
        t = self.trainer(); pool=t._opponent_pool
        for probs in ([float('nan')],[-1],[0],[.5,.5]):
            with patch.object(pool._policies[0], 'probabilities', return_value=probs):
                with self.assertRaises(ValueError): pool.action(0,{},[{'type':'wait'}],np.random.default_rng(1))

    def test_explicit_import_provenance_and_pending_rejection(self):
        t = NeuralTrainer.load(self.path,self.snapshot)
        before=state_digest(t.policy.checkpoint(arrays=True))
        audit=t.import_opponent_pool_for_new_run(self.pool,parent_checkpoint_sha256=file_sha256(self.path))
        self.assertIsNone(audit['previous_pool']); self.assertEqual(audit['at_episode'],2)
        self.assertEqual(before,state_digest(t.policy.checkpoint(arrays=True)))
        child=self.root/'child.npz';t.save(child)
        restored=NeuralTrainer.load(child,self.snapshot)
        self.assertEqual(restored.checkpoint()['opponent_pool_import'],audit)
        t.train(1)
        with self.assertRaisesRegex(ValueError,'completed batch'):
            t.import_opponent_pool_for_new_run(None,parent_checkpoint_sha256=file_sha256(self.path))

    def test_cloud_import_and_resume_pool_binding(self):
        report=self.root/'training.json'
        report.write_text(json.dumps({'training':self.source_rows,'training_summary':summary(self.source_rows),
            'training_elapsed_seconds':0,'episodes_attempted':2,'episodes_completed':2,
            'optimizer_updates':self.source.policy.steps}))
        args=cloud_train.parser().parse_args(['--snapshot',str(self.snapshot),'--output',str(self.root/'runs'),
            '--run-id','child','--episodes','4','--initialize-from',str(self.path),
            '--parent-training-report',str(report),'--checkpoint-format','npz'])
        config={**self.config,'allow_real':True,'train_seed_start':0,'scoring_mode':'verified','opponent_pool':self.pool}
        stop=self.root/'stop';stop.touch();args.stop_file=stop
        with patch.object(cloud_train,'requested_config',return_value=config):
            self.assertEqual(cloud_train.run(args),130)
        directory=self.root/'runs/child'
        manifest=json.loads((directory/'manifest.json').read_text())
        self.assertEqual(manifest['opponent_pool']['config'],self.pool)
        self.assertIn('opponent_pool_import',manifest['initialization'])
        args.resume=True;args.initialize_from=args.parent_training_report=None
        with patch.object(cloud_train,'requested_config',return_value=config):
            self.assertEqual(cloud_train.run(args),130)
        changed=copy.deepcopy(config);changed['opponent_pool']=describe_pool([self.path])
        with patch.object(cloud_train,'requested_config',return_value=changed),self.assertRaisesRegex(ValueError,'configuration'):
            cloud_train.run(args)


if __name__ == '__main__': unittest.main()
