"""Bounded immutable NPZ opponents. No environment or learner references enter policies."""
import copy
import hashlib
from pathlib import Path

import numpy as np

from engine import scoring
from .compact_checkpoint import read_checkpoint, state_digest
from .neural_policy import NeuralPolicy

MAX_OPPONENTS = 8
MAX_POOL_BYTES = 512 * 1024 * 1024


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def describe_pool(paths, *, include_uniform=False):
    """Seal ordered membership by content. Order defines equal sampling weights."""
    paths = list(paths)
    if not paths or len(paths) + int(include_uniform) > MAX_OPPONENTS:
        raise ValueError('frozen pool requires 1..8 entries including optional uniform')
    resolved = [Path(p).resolve() for p in paths]
    if len(set(resolved)) != len(resolved) or any(p.suffix.lower() != '.npz' for p in resolved):
        raise ValueError('frozen opponents require unique NPZ checkpoint paths')
    if sum(p.stat().st_size for p in resolved) > MAX_POOL_BYTES:
        raise ValueError('frozen opponent pool exceeds byte limit')
    entries = [{'kind': 'checkpoint', 'path': str(p), 'sha256': file_sha256(p)} for p in resolved]
    if len({e['sha256'] for e in entries}) != len(entries):
        raise ValueError('duplicate frozen checkpoint content')
    if include_uniform:
        entries.append({'kind': 'uniform'})
    return {'version': 1, 'sampling': 'equal_per_episode_v1', 'entries': entries}


class FrozenPool:
    def __init__(self, config, *, snapshot, episode_config, observation_schema, scoring_mode):
        if (not isinstance(config, dict) or set(config) != {'version', 'sampling', 'entries'} or
                config['version'] != 1 or config['sampling'] != 'equal_per_episode_v1'):
            raise ValueError('invalid frozen pool contract')
        entries = config['entries']
        if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_OPPONENTS:
            raise ValueError('frozen pool must contain 1..8 entries')
        paths, uniform = [], 0
        for entry in entries:
            if entry == {'kind': 'uniform'}:
                uniform += 1
            elif (isinstance(entry, dict) and set(entry) == {'kind', 'path', 'sha256'}
                    and entry['kind'] == 'checkpoint'):
                paths.append(entry['path'])
            else:
                raise ValueError('invalid frozen pool entry')
        if uniform > 1 or describe_pool(paths, include_uniform=bool(uniform)) != config:
            raise ValueError('frozen pool source changed; explicit new-run import required')
        if observation_schema not in (2, 3, 4):
            raise ValueError('frozen training requires public observation schema 2..4')
        self._config = copy.deepcopy(config)
        self._policies = []
        self._contracts = []
        total = 0
        for entry in entries:
            if entry['kind'] == 'uniform':
                self._policies.append(None)
                self._contracts.append({'kind': 'uniform'})
                continue
            data = read_checkpoint(entry['path'])
            if (data.get('schema_version') != 1 or data.get('snapshot') != snapshot or
                    data.get('episode_config') != episode_config or
                    data.get('observation_schema', 1) != observation_schema or
                    data.get('scoring_version') != scoring.SCORING_VERSION or
                    data.get('scoring_mode', 'verified') != scoring_mode):
                raise ValueError('frozen opponent snapshot/schema/scoring/episode contract differs')
            if data.get('algorithm') not in ('neural_terminal_reinforce_vs_random',
                    'neural_terminal_ppo_clip_vs_random', 'neural_terminal_reinforce_vs_frozen_pool',
                    'neural_terminal_ppo_clip_vs_frozen_pool'):
                raise ValueError('unsupported frozen opponent checkpoint')
            policy = NeuralPolicy.from_checkpoint(data['policy'])
            if policy.steps == 0:
                raise ValueError('frozen opponent must be an already trained policy')
            representation = {2: 'public_v2_split', 3: 'public_v3_spatial', 4: 'public_v4_contacts'}[observation_schema]
            if policy.config['representation'] != representation:
                raise ValueError('frozen opponent representation differs from schema')
            for group in (policy.params, policy.m, policy.v):
                for value in group.values():
                    total += value.nbytes
                    value.setflags(write=False)
            if total > MAX_POOL_BYTES:
                raise ValueError('frozen opponent tensors exceed byte limit')
            self._policies.append(policy)
            self._contracts.append({**entry, 'policy_config': copy.deepcopy(policy.config),
                'observation_schema': observation_schema, 'scoring_version': scoring.SCORING_VERSION,
                'scoring_mode': scoring_mode, 'snapshot': copy.deepcopy(snapshot),
                'episode_config': copy.deepcopy(episode_config),
                'state_sha256': state_digest(policy.checkpoint(arrays=True))})
            del data
        self.verify_sources()

    @property
    def config(self):
        return copy.deepcopy(self._config)

    @property
    def manifest(self):
        return {'config': self.config, 'opponents': copy.deepcopy(self._contracts)}

    def verify_sources(self):
        for entry in self._config['entries']:
            if entry['kind'] == 'checkpoint' and file_sha256(entry['path']) != entry['sha256']:
                raise ValueError('frozen opponent source changed; explicit new-run import required')

    def state_digest(self):
        return state_digest([p.checkpoint(arrays=True) if p else None for p in self._policies])

    def select(self, seed):
        # Selection owns a fresh stream; never advances learner or action RNGs.
        return int(np.random.default_rng(seed).integers(len(self._policies)))

    def is_uniform(self, index):
        return self._policies[index] is None

    def action(self, index, observation, candidates, rng):
        policy = self._policies[index]
        if policy is None:
            raise ValueError('uniform entry uses legal-action sampling')
        probs = np.asarray(policy.probabilities(copy.deepcopy(observation), copy.deepcopy(candidates)))
        if (probs.shape != (len(candidates),) or not np.isfinite(probs).all() or
                (probs < 0).any() or not np.isclose(probs.sum(), 1)):
            raise ValueError('invalid frozen opponent probabilities')
        return copy.deepcopy(candidates[int(rng.choice(len(candidates), p=probs))])
