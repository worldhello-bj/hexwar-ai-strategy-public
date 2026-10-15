import copy,json,tempfile,unittest
from pathlib import Path
from env.match_eval import write_synthetic_snapshot
from env.neural_train import NeuralTrainer
from tools.migrate_neural_backend import migrate

from tests.test_sparse_backend import HAS_SCIPY, SCIPY_REASON

@unittest.skipUnless(HAS_SCIPY, SCIPY_REASON)
class MigrationTests(unittest.TestCase):
    def test_exact_state_and_guarded_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);snapshot=write_synthetic_snapshot(root/'snapshot')
            trainer=NeuralTrainer(snapshot,batch_episodes=2,episode_config={'max_seconds':10},policy_config={'buckets':8,'hidden_dim':2,'update_mode':'ppo_clip'})
            trainer.train(1);pending=root/'pending.json';trainer.save(pending)
            with self.assertRaisesRegex(ValueError,'empty pending'):migrate(pending,root/'blocked.json',snapshot)
            self.assertFalse((root/'blocked.json').exists())
            trainer.train(1);source=root/'source.json';trainer.save(source);original=source.read_bytes()
            target=root/'csr.json';audit=migrate(source,target,snapshot)
            self.assertEqual(source.read_bytes(),original)
            restored=NeuralTrainer.load(target,snapshot)
            self.assertEqual(restored.policy.config['projection_backend'],'scipy_csr')
            for key in ('policy','initial_policy'):
                for field in ('params','m','v','rng','steps'):
                    self.assertEqual(json.loads(original)[key][field],restored.checkpoint()[key][field])
            self.assertTrue(target.with_suffix('.json.migration.json').exists())
            self.assertTrue(audit['pending_empty'])
            with self.assertRaises(ValueError):migrate(source,target,snapshot)
            with self.assertRaises(ValueError):migrate(source,source,snapshot)
            with self.assertRaisesRegex(ValueError,'numpy'):migrate(target,root/'already.json',snapshot)

if __name__=='__main__':unittest.main()
