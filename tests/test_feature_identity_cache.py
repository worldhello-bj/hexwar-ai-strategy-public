import unittest
import numpy as np
from env.neural_policy import NeuralPolicy
from tests.fixtures.pre_ppo_neural_policy import NeuralPolicy as Reference

class IdentityCacheTests(unittest.TestCase):
    def test_exact_features_mutable_inputs_and_config_invalidation(self):
        p,q=NeuralPolicy(buckets=32,hidden_dim=4),Reference(buckets=32,hidden_dim=4)
        value={'nested':[True,None,'中文',-0.0,{'x':5.2}]}
        for seed in (0,5,{'arbitrary':'JSON-compatible legacy seed'}):
            p.config['hash_seed']=q.config['hash_seed']=seed
            for number in (2,-5.5,0,1000):
                value['nested'][-1]['x']=number
                a,b=p._features(value),q._features(value)
                np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        self.assertEqual(p.checkpoint(),q.checkpoint())
    def test_public_v2_exact_probabilities_samples_and_checkpoint(self):
        config = dict(seed=23, buckets=32, hidden_dim=4, representation="public_v2_split")
        p, q = NeuralPolicy(**config), Reference(**config)
        observation = {"schema_version": 2, "time": 0,
                       "board": {"cells": [{"x": 1, "terrain": "road"}]}}
        candidates = [{"action": {"type": "wait", "dt": 10}, "public_context": {}},
                      {"action": {"type": "move", "to": 2}, "public_context": {"x": 2}}]
        for tick in range(5):
            observation["time"] = tick
            np.testing.assert_array_equal(p.probabilities(observation, candidates),
                                          q.probabilities(observation, candidates))
            a, d = p.sample(observation, candidates)
            b, e = q.sample(observation, candidates)
            self.assertEqual(a, b)
            self.assertEqual(d.to_dict(), e.to_dict())
            self.assertEqual(p.checkpoint(), q.checkpoint())

    def test_cache_is_bounded_and_eviction_does_not_change_features(self):
        p,q=NeuralPolicy(buckets=8,hidden_dim=2),Reference(buckets=8,hidden_dim=2)
        first=p._features('original')
        for i in range(33000):p._features(str(i))
        self.assertLessEqual(len(p._feature_hash_cache),32768)
        actual=p._features('original');expected=q._features('original')
        for a,b,c in zip(actual,expected,first):
            np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a,c)

if __name__=='__main__':unittest.main()
