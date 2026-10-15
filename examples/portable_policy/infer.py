"""Infer from a detached, own-seat schema-3 public observation and legal views."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from hexwar_policy.inference_format import load_weights

class GamePolicy:
    """A cooperative API boundary, not a security sandbox or legality oracle.

    Host must provide only the acting seat's public observation and current
    advertised legal candidate views from SpatialObservationAdapter.
    No engine, opponent, snapshot, trainer or hidden-state object is accepted.
    """
    def __init__(self, model, *, seed=0):
        self._policy = load_weights(model)
        self._rng = np.random.default_rng(seed)

    def choose(self, observation, candidates):
        observation, candidates = json.loads(json.dumps(
            [observation, candidates], allow_nan=False))
        if observation.get('schema_version') != 3 or observation.get('side') not in ('BLUE', 'RED'):
            raise ValueError('Expected an own-seat schema-3 public observation')
        if not candidates:
            raise ValueError('No legal candidates for this seat')
        probs = np.asarray(self._policy.probabilities(observation, candidates))
        if not np.isfinite(probs).all() or (probs < 0).any() or not np.isclose(probs.sum(), 1):
            raise ValueError('Invalid policy probabilities')
        index = int(self._rng.choice(len(candidates), p=probs))
        return {'index': index, 'action': copy.deepcopy(candidates[index]['action']),
                'probabilities': probs.tolist()}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('public_input', help='JSON with observation and candidates keys')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    data = json.loads(Path(args.public_input).read_text())
    print(json.dumps(GamePolicy(args.model, seed=args.seed).choose(
        data['observation'], data['candidates']), allow_nan=False))

if __name__ == '__main__':
    main()
