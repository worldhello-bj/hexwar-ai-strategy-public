"""Evidence-based trajectory diagnostics; requested routes are never movement."""
import copy
import io
import unittest

from tools.review_training_replay import review_replay, write_html


def observation(pos, score=0, loaded=None):
    return {'time': 0, 'own': {'u': {'id': 'u', 'typeid': '1', 'pos': str(pos), 'loaded_into': loaded}},
            'score': {'own': score, 'enemy': 0}}


def fixture():
    initial = observation(0)
    actions = [{'type': 'move', 'unit': 'u', 'path': [0, 2]},
               {'type': 'move', 'unit': 'u', 'path': [2, 0]},
               {'type': 'move', 'unit': 'u', 'path': [0, 4]},
               {'type': 'wait'}, {'type': 'wait'},
               {'type': 'load', 'unit': 'u', 'carrier': 'c'},
               {'type': 'unload', 'unit': 'c'}]
    after = [observation(2), observation(0), observation(0), observation(0),
             observation(0), observation(0, 2, 'c'), observation(0, 3)]
    decisions, steps = [], []
    before = initial
    for i, (action, obs) in enumerate(zip(actions, after)):
        decisions.append({'step': i, 'observation': before, 'selected_action': action,
                          'candidates': [{'action': {'type': 'move', 'unit': 'u'}}]})
        steps.append({'after_step': i+1, 'observation': obs,
                      'actor_is_learner': True, 'learner_decision_index': i})
        before = obs
    return {'initial_observation': initial, 'decisions': decisions, 'observed_steps': steps,
            'final_observation': after[-1], 'ending': {'steps': len(steps)}, 'seed': 1,
            'learner_side': 'BLUE'}


class ReviewTests(unittest.TestCase):
    def test_requests_actual_distance_cycles_wait_and_transport(self):
        report = review_replay(fixture())
        self.assertEqual(report['move_requests'], 3)
        self.assertEqual(report['unique_observed_displaced_units'], 1)
        self.assertEqual(report['observed_displaced_eligible_fraction'], 1)
        self.assertEqual(report['move_requests_without_immediate_observed_displacement'], 1)
        self.assertEqual(report['max_consecutive_learner_wait_decisions'], 2)
        unit = report['units'][0]
        self.assertEqual(unit['observed_cumulative_hex_distance_lower_bound'], 2)
        self.assertEqual(unit['first_to_last_observed_hex_distance'], 0)
        self.assertEqual(unit['return_to_previously_observed_hex_count'], 1)
        self.assertEqual(report['transport_public_score_changes']['load']['sum_positive_delta'], 2)
        self.assertEqual(report['transport_public_score_changes']['unload']['sum_positive_delta'], 1)
        self.assertEqual(len(report['transport_events'][0]['observed_state_transitions']), 1)
        self.assertIsNone(report['invalid_actions'])
        self.assertIsNone(report['proven_noop_actions'])

    def test_static_candidates_excluded_and_actual_static_movement_flagged(self):
        replay = fixture()
        observations = [replay['initial_observation']] + [f['observation'] for f in replay['observed_steps']]
        for index, obs in enumerate(observations):
            obs['own']['bunker'] = {'id': 'bunker', 'typeid': 4, 'pos': 0 if index == 0 else 2}
            obs['own']['unknown'] = {'id': 'unknown', 'pos': 0}
        for decision in replay['decisions']:
            decision['candidates'] += [{'action': {'type': 'move', 'unit': 'bunker'}},
                                       {'action': {'type': 'move', 'unit': 'unknown'}}]
        report = review_replay(replay)
        self.assertEqual(report['unique_units_with_advertised_move_candidate'], 3)
        self.assertEqual(report['unique_mobile_units_with_advertised_move_candidate'], 1)
        self.assertEqual(report['observed_displaced_eligible_units'], 1)
        self.assertEqual(report['observed_displaced_eligible_fraction'], 1)
        self.assertEqual(report['unique_observed_displaced_units'], 2)
        self.assertEqual(report['advertised_static_type4_units'], ['bunker'])
        self.assertEqual(report['advertised_units_missing_observed_typeid'], ['unknown'])
        self.assertTrue(report['has_static_movement_violation'])
        self.assertEqual(report['static_movement_violations'][0]['from_hex'], 0)
        self.assertEqual(report['static_movement_violations'][0]['to_hex'], 2)
        self.assertEqual(report['ending'], replay['ending'])
        stream = io.StringIO()
        write_html(report, stream)
        self.assertIn('not valid evidence of playing strength', stream.getvalue())

    def test_requested_path_without_movement_is_not_distance(self):
        replay = fixture()
        for frame in replay['observed_steps']:
            frame['observation']['own']['u']['pos'] = '0'
        report = review_replay(replay)
        self.assertEqual(report['unique_observed_displaced_units'], 0)
        self.assertEqual(report['units'][0]['observed_cumulative_hex_distance_lower_bound'], 0)
        self.assertEqual(report['move_requests_without_immediate_observed_displacement'], 3)

    def test_legacy_and_private_score_are_not_fabricated(self):
        replay = fixture()
        del replay['observed_steps']
        report = review_replay(replay)
        self.assertIsNone(report['move_requests_without_immediate_observed_displacement'])
        self.assertIsNone(report['transport_public_score_changes']['load']['sum_positive_delta'])
        replay = fixture()
        for frame in replay['observed_steps']:
            frame['observation'].pop('score')
        report = review_replay(replay)
        self.assertIsNone(report['transport_public_score_changes']['load']['sum_positive_delta'])

    def test_legacy_list_units_and_gaps_do_not_invent_connected_distance(self):
        replay = fixture()
        replay['observed_steps'][0]['observation']['own'] = []
        replay['observed_steps'][1]['observation']['own'] = [{'id': 'u', 'pos': 10000}]
        report = review_replay(replay)
        self.assertEqual(report['units'][0]['observation_gaps'], 1)
        self.assertFalse(report['units'][0]['path'][1]['connected_to_previous'])

    def test_html_contains_real_coordinates_and_escapes_unit_names(self):
        replay = fixture()
        report = review_replay(replay)
        report['units'][0]['unit'] = '<script>bad()</script>'
        stream = io.StringIO()
        write_html(report, stream)
        self.assertIn('viewBox=', stream.getvalue())
        self.assertIn('hex 2', stream.getvalue())
        self.assertIn('&lt;script&gt;', stream.getvalue())
        self.assertNotIn('<script>', stream.getvalue())

    def test_analysis_does_not_mutate_replay(self):
        replay = fixture()
        before = copy.deepcopy(replay)
        review_replay(replay)
        self.assertEqual(replay, before)


if __name__ == '__main__':
    unittest.main()
