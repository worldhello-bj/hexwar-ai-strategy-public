import unittest
import numpy as np
from env.neural_policy import NeuralPolicy
import tests.test_neural_ppo as ppo_tests

try:
    import scipy.sparse
except ModuleNotFoundError as exc:
    if exc.name != 'scipy' and not exc.name.startswith('scipy.'):
        raise
    HAS_SCIPY = False
else:
    HAS_SCIPY = True

SCIPY_REASON = 'optional CSR tests require requirements-sparse.txt'

@unittest.skipUnless(HAS_SCIPY, SCIPY_REASON)
class CSRClippedPolicyTests(ppo_tests.ClippedPolicyTests):
    def policy(self, **kwargs):
        return NeuralPolicy(buckets=8,hidden_dim=2,update_mode='ppo_clip',projection_backend='scipy_csr',**kwargs)

@unittest.skipUnless(HAS_SCIPY, SCIPY_REASON)
class BackendEqualityTests(unittest.TestCase):
    def test_forward_gradients_updates_and_exact_resume(self):
        config=dict(seed=17,buckets=32,hidden_dim=4,update_mode='ppo_clip',update_epochs=4)
        p,q=NeuralPolicy(**config),NeuralPolicy(**config,projection_backend='scipy_csr')
        d=p._decision({'values':[1,2,3]},[{'a':1},{'a':2},{'a':3}],1)
        examples=[(d,14.),(d,-3.)]
        for a,b in zip(p._forward(d),q._forward(d)):
            np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-14)
        loss,g=p._loss_gradients(examples);other,h=q._loss_gradients(examples)
        self.assertAlmostEqual(loss,other,places=12)
        for k in g:np.testing.assert_allclose(g[k],h[k],rtol=1e-12,atol=1e-14)
        p.update_batch([([d],14.),([d],-3.)]);q.update_batch([([d],14.),([d],-3.)])
        for group in ('params','m','v'):
            for k in getattr(p,group):np.testing.assert_allclose(getattr(p,group)[k],getattr(q,group)[k],rtol=1e-10,atol=1e-11)
        self.assertEqual(p.rng.bit_generator.state,q.rng.bit_generator.state)
        restored=NeuralPolicy.from_checkpoint(q.checkpoint())
        self.assertEqual(q.checkpoint(),restored.checkpoint())
        q.update_batch([([d],8.)]);restored.update_batch([([d],8.)])
        self.assertEqual(q.checkpoint(),restored.checkpoint())


class BackendConfigTests(unittest.TestCase):
    def test_invalid_backend_and_default_checkpoint_compatibility(self):
        with self.assertRaises(ValueError):NeuralPolicy(projection_backend='other')
        p=NeuralPolicy(buckets=8,hidden_dim=2)
        self.assertNotIn('projection_backend',p.checkpoint()['config'])
    @unittest.skipUnless(HAS_SCIPY, SCIPY_REASON)
    def test_explicit_csr_checkpoint_config(self):
        q=NeuralPolicy(buckets=8,hidden_dim=2,projection_backend='scipy_csr')
        self.assertEqual(q.checkpoint()['config']['projection_backend'],'scipy_csr')
    def test_entrypoint_opt_in(self):
        from tools.cloud_train import parser
        from train.run_train import to_argv,ALLOWED
        self.assertEqual(parser().parse_args([]).projection_backend,'numpy')
        self.assertEqual(parser().parse_args(['--projection-backend','scipy_csr']).projection_backend,'scipy_csr')
        self.assertIn('projection_backend',ALLOWED['ppo'])
        self.assertEqual(to_argv({'projection_backend':'scipy_csr'}),['--projection-backend','scipy_csr'])

if __name__=='__main__':unittest.main()
