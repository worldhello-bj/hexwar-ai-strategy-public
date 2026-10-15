"""Synthetic inference contract check; no trained models or game data required."""
import copy
from pathlib import Path
import tempfile
import numpy as np
from hexwar_policy.neural_policy import NeuralPolicy
from hexwar_policy.inference_format import save_weights, load_weights
from infer import GamePolicy


def main():
    policy = NeuralPolicy(seed=7, buckets=16, hidden_dim=3,
                          representation='public_v3_spatial',
                          baseline_offset='public_own_score')
    observation = {'schema_version': 3, 'side': 'BLUE', 'board': {},
                   'own': [], 'enemy_visible': [], 'score': {'own': 0}}
    candidates = [{'action': {'type': 'wait', 'synthetic_id': i},
                   'public_context': {'objective_relations': []}} for i in range(2)]
    before = copy.deepcopy((observation, candidates))
    expected = policy.probabilities(observation, candidates)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'synthetic.npz'
        save_weights(path, policy)
        loaded = load_weights(path)
        assert loaded.config == policy.config
        for key in policy.params:
            np.testing.assert_array_equal(loaded.params[key], policy.params[key])
        np.testing.assert_array_equal(loaded.probabilities(observation, candidates), expected)
        a, b = GamePolicy(path, seed=42), GamePolicy(path, seed=42)
        for _ in range(8):
            first, second = a.choose(observation, candidates), b.choose(observation, candidates)
            assert first == second
            assert first['action'] == candidates[first['index']]['action']
            np.testing.assert_array_equal(first['probabilities'], expected)
        assert (observation, candidates) == before
        for invalid_obs, invalid_candidates in [({**observation, 'side': 'UNKNOWN'}, candidates),
                                                 (observation, [])]:
            try:
                a.choose(invalid_obs, invalid_candidates)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid input accepted')
    print('Synthetic portable-policy smoke test passed')


if __name__ == '__main__':
    main()
