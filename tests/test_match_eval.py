"""Software-only regressions, using original abstract tokens without weapons."""
import json
from pathlib import Path
import random
import tempfile
import unittest

from env.match_eval import (evaluate_paired, random_policy, run_match,
                            snapshot_manifest, summarize, write_synthetic_snapshot)


def wait_policy(seed):
    def choose(observation, actions):
        return next(a for a in actions if a['type'] == 'wait')
    return choose


class MatchEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = write_synthetic_snapshot(Path(self.tmp.name) / 'synthetic')

    def match(self, **kwargs):
        return run_match(self.path, {'BLUE': wait_policy(1), 'RED': wait_policy(1)}, **kwargs)

    def test_uniform_random_paired_smoke_reproducible(self):
        random.seed(997)
        state = random.getstate()
        first = evaluate_paired(self.path, random_policy, random_policy,
                                seeds=(3, 9), max_seconds=30, max_turns=12)
        second = evaluate_paired(self.path, random_policy, random_policy,
                                 seeds=(3, 9), max_seconds=30, max_turns=12)
        self.assertEqual(first, second)
        self.assertEqual(state, random.getstate())
        self.assertEqual(len(first['records']), 4)
        self.assertEqual(first['records'][0]['seats'], {'BLUE': 'A', 'RED': 'B'})
        self.assertEqual(first['records'][1]['seats'], {'BLUE': 'B', 'RED': 'A'})
        json.dumps(first, allow_nan=False)

    def test_one_clock_and_alternating_seats(self):
        record = self.match(max_seconds=40)
        self.assertEqual([r['side'] for r in record['trace']], ['BLUE', 'RED', 'BLUE', 'RED'])
        self.assertEqual([r['time_before'] for r in record['trace']], [0, 10, 20, 30])
        self.assertEqual(record['seconds'], 40)
        self.assertTrue(record['outcome']['truncated'])
        self.assertFalse(record['outcome']['terminated'])
        self.assertIsNone(record['outcome']['winner'])
        self.assertEqual(record['outcome']['reason'], 'time_limit')

    def test_policy_inputs_detached_and_seat_specific(self):
        seen = []
        def factory(seed):
            def choose(obs, actions):
                seen.append((obs['side'], [u['id'] for u in obs['own']]))
                action = next(a.copy() for a in actions if a['type'] == 'wait')
                obs['own'][0]['blood'] = -999
                actions.clear()
                return action
            return choose
        record = run_match(self.path, {'BLUE': factory(0), 'RED': factory(0)}, max_seconds=20)
        self.assertEqual(seen, [('BLUE', ['blue']), ('RED', ['red'])])
        self.assertEqual(record['final_observations']['BLUE']['own'][0]['blood'], 4)
        self.assertTrue(record['trace'][0]['legal_actions'])

    def test_zero_budgets_and_invalid_parameters(self):
        self.assertEqual(self.match(max_turns=0)['outcome']['reason'], 'turn_limit')
        self.assertEqual(self.match(max_seconds=0)['outcome']['reason'], 'time_limit')
        for kwargs in ({'max_turns': -1}, {'move_radius': True}, {'seed': -1}):
            with self.assertRaises(ValueError):
                self.match(**kwargs)
        p = wait_policy(0)
        with self.assertRaises(ValueError):
            run_match(self.path, {'BLUE': p, 'RED': p})
        with self.assertRaises(ValueError):
            evaluate_paired(self.path, random_policy, random_policy, seeds=(1, 1))

    def test_policy_exception_and_invalid_action_not_counted_as_wins(self):
        def bad(obs, actions):
            raise RuntimeError('deliberate fixture failure')
        for policy, reason in ((bad, 'policy_exception'),
                               (lambda o, a: {'type': 'not-an-action'}, 'policy_action_not_legal')):
            record = run_match(self.path, {'BLUE': policy, 'RED': wait_policy(0)})
            self.assertEqual(record['outcome']['reason'], reason)
            self.assertIsNone(record['outcome']['winner'])
            self.assertEqual(record['seconds'], 0)

    def test_snapshot_hash_changes_and_requires_provenance(self):
        before = snapshot_manifest(self.path)['sha256']
        file = self.path / 'units.json'
        file.write_text(file.read_text() + '\n')
        self.assertNotEqual(before, snapshot_manifest(self.path)['sha256'])
        (self.path / 'evaluation_provenance.json').write_text('{"origin":"unknown"}')
        with self.assertRaises(ValueError):
            self.match()

    def test_natural_terminal_and_truncation_are_distinct(self):
        file = self.path / 'units.json'
        units = json.loads(file.read_text())
        file.write_text(json.dumps(units[:1]))
        record = self.match(max_seconds=0)
        self.assertTrue(record['outcome']['terminated'])
        self.assertFalse(record['outcome']['truncated'])
        self.assertEqual(record['outcome']['winner'], 'BLUE')
        self.assertEqual(record['turns'], 0)
        file.write_text('[]')
        self.assertEqual(self.match()['outcome']['reason'], 'both_sides_empty')

    def test_summary_uses_paired_clusters_excludes_incomplete(self):
        def row(pair, winner, terminal=True):
            return {'pair_id': pair, 'seats': {'BLUE': 'A', 'RED': 'B'},
                    'outcome': {'terminated': terminal, 'winner': winner}}
        result = summarize([row(0, 'BLUE'), row(0, None), row(1, 'RED'), row(1, None, False)])
        self.assertEqual([result[k] for k in ('wins', 'draws', 'losses', 'incomplete')], [1, 1, 1, 1])
        self.assertEqual(result['complete_pairs'], 1)
        self.assertEqual(result['paired_statistics']['win_fraction']['mean'], 0.5)
        self.assertEqual(result['paired_statistics']['score_fraction']['mean'], 0.75)
        self.assertEqual(result['paired_statistics']['score_fraction']['confidence_interval_95'], [0, 1])
        self.assertIsNone(summarize([])['paired_statistics']['win_fraction']['mean'])

    def test_fresh_factories_each_leg(self):
        calls = []
        def factory(seed):
            calls.append(seed)
            return wait_policy(seed)
        evaluate_paired(self.path, factory, factory, seeds=(7, 11), max_turns=0)
        self.assertEqual(calls, [7, 7, 7, 7, 11, 11, 11, 11])


if __name__ == '__main__':
    unittest.main()
