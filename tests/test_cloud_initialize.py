import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from env.match_eval import write_synthetic_snapshot
from env.neural_train import NeuralTrainer,summary
from tools import cloud_train
from tools.migrate_neural_backend import migrate

from tests.test_sparse_backend import HAS_SCIPY, SCIPY_REASON

@unittest.skipUnless(HAS_SCIPY, SCIPY_REASON)
class InitializeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.snapshot=write_synthetic_snapshot(self.root/'snapshot')
        self.config=dict(seed=17,train_seed_start=10,batch_episodes=2,heldout_seeds=[100],
            observation_schema=2,allow_real=True,scoring_mode='verified',
            episode_config={'max_seconds':20,'max_steps':100,'move_radius':1},
            policy_config={'buckets':16,'hidden_dim':3,'learning_rate':.001,
                'baseline_offset':'prior_batch_mean','update_mode':'ppo_clip'})
        t=NeuralTrainer(self.snapshot,**self.config);records=t.train(2)
        self.original=self.root/'numpy.json';t.save(self.original)
        self.parent=self.root/'csr.json';migrate(self.original,self.parent,self.snapshot)
        self.config['policy_config']['projection_backend']='scipy_csr'
        self.report=self.root/'training.json'
        self.report.write_text(json.dumps({'training':records,'training_summary':summary(records),
             'episodes_attempted':2,'episodes_completed':2,'optimizer_updates':t.policy.steps,
             'training_elapsed_seconds':1.25}))
        self.args=cloud_train.parser().parse_args(['--snapshot',str(self.snapshot),'--output',str(self.root/'output'),
             '--run-id','child','--episodes','4','--seed','17','--train-seed-start','10','--batch-episodes','2',
             '--eval-seeds','100','--projection-backend','scipy_csr','--initialize-from',str(self.parent),
             '--parent-training-report',str(self.report)])
    def test_import_preserves_state_journal_and_resume_continuation(self):
        before=self.parent.read_bytes();journal=self.report.read_bytes()
        loaded,rows,elapsed,audit=cloud_train.import_completed_checkpoint(self.args,self.config)
        self.assertEqual(loaded.checkpoint(),json.loads(before));self.assertEqual(elapsed,1.25)
        self.assertEqual(rows,json.loads(journal)['training'])
        self.assertEqual(audit['parent_checkpoint_sha256'],cloud_train.file_sha256(self.parent))
        self.assertEqual(audit['parent_training_report_sha256'],cloud_train.file_sha256(self.report))
        stop=self.root/'stop';stop.touch();self.args.stop_file=stop
        with patch.object(cloud_train,'requested_config',return_value=self.config):
            self.assertEqual(cloud_train.run(self.args),130)
        child=self.root/'output/child'
        self.assertEqual(json.loads((child/'checkpoint.json').read_text()),json.loads(before))
        self.assertEqual(json.loads((child/'training.json').read_text())['training'],rows)
        self.assertEqual(json.loads((child/'manifest.json').read_text())['initialization'],audit)
        self.assertEqual(json.loads((child/'import-audit.json').read_text()),audit)
        with patch.object(cloud_train,'requested_config',return_value=self.config),self.assertRaises(FileExistsError):
            cloud_train.run(self.args)
        self.assertEqual(self.parent.read_bytes(),before);self.assertEqual(self.report.read_bytes(),journal)
        expected=NeuralTrainer.load(self.parent,self.snapshot,allow_real=True);expected.train(2)
        self.args.resume=True;self.args.initialize_from=self.args.parent_training_report=None;stop.unlink()
        original_train=NeuralTrainer.train
        def train_and_stop(trainer,count):
            result=original_train(trainer,count)
            if trainer.episodes_attempted==4:stop.touch()
            return result
        with patch.object(cloud_train,'requested_config',return_value=self.config),patch.object(NeuralTrainer,'train',train_and_stop):
            self.assertEqual(cloud_train.run(self.args),130)
        self.assertEqual(json.loads((child/'checkpoint.json').read_text()),expected.checkpoint())
        final_rows=json.loads((child/'training.json').read_text())['training']
        self.assertEqual(len(final_rows),4);self.assertEqual(final_rows[:2],rows)
        self.assertEqual(self.parent.read_bytes(),before);self.assertEqual(self.report.read_bytes(),journal)
    def test_config_and_budget_mismatches_fail_before_creation(self):
        for key,value in [('seed',18),('train_seed_start',11),('batch_episodes',4),
                          ('heldout_seeds',[101]),('observation_schema',3),('scoring_mode','platform')]:
            config=copy.deepcopy(self.config);config[key]=value
            with self.subTest(key=key),patch.object(cloud_train,'requested_config',return_value=config),self.assertRaises(ValueError):
                cloud_train.run(self.args)
            self.assertFalse((self.root/'output/child').exists())
        for group,key,value in [('episode_config','move_radius',3),('episode_config','max_seconds',1800),
                    ('episode_config','max_steps',1000),('policy_config','learning_rate',.01),
                    ('policy_config','projection_backend','numpy')]:
            config=copy.deepcopy(self.config);config[group][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'configuration'):
                cloud_train.import_completed_checkpoint(self.args,config)
        self.args.episodes=1
        with self.assertRaisesRegex(ValueError,'cumulative'):cloud_train.import_completed_checkpoint(self.args,self.config)
    def test_bad_journal_pending_and_migration_audit(self):
        saved=json.loads(self.report.read_text())
        for change in ('count','seed','summary','elapsed','version'):
            data=copy.deepcopy(saved)
            if change=='count':data['training']=data['training'][:-1]
            if change=='seed':data['training'][0]['seed']=999
            if change=='summary':data['training_summary']['games']=99
            if change=='elapsed':data['training_elapsed_seconds']=-1
            if change=='version':data['training'][0]['status']['scoring_version']=7
            self.report.write_text(json.dumps(data))
            with self.subTest(change=change),self.assertRaises(ValueError):cloud_train.import_completed_checkpoint(self.args,self.config)
        self.report.write_text(json.dumps(saved))
        audit_path=self.parent.with_suffix('.json.migration.json');audit=json.loads(audit_path.read_text())
        audit['output_checkpoint_sha256']='incorrect';audit_path.write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError,'migration audit'):cloud_train.import_completed_checkpoint(self.args,self.config)
        t=NeuralTrainer.load(self.parent,self.snapshot,allow_real=True);t.train(1)
        pending=self.root/'pending.json';t.save(pending);self.args.initialize_from=pending
        with self.assertRaisesRegex(ValueError,'empty pending'):cloud_train.import_completed_checkpoint(self.args,self.config)
    def test_import_flag_pairs_and_resume_exclusion(self):
        self.args.parent_training_report=None
        with self.assertRaisesRegex(ValueError,'requires'):cloud_train.run(self.args)
        self.args.parent_training_report=self.report;self.args.resume=True
        with self.assertRaisesRegex(ValueError,'combined'):cloud_train.run(self.args)

if __name__=='__main__':unittest.main()
