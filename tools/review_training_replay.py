#!/usr/bin/env python3
"""Audit a frozen learner-seat replay using only exported observations.

Observed distances are sums of hex distances between sampled positions, not
engine path lengths. No opponent private state or invalid-action reasons exist
in this interface. This diagnoses evidence, not tactical optimality.
"""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.geometry import distance, split_mapid


def own(observation):
    units = observation.get('own', {})
    return {str(u['id']): u for u in (units.values() if isinstance(units, dict) else units)}


def position(unit):
    value = unit.get('pos')
    if value is None:
        value = unit.get('position', {}).get('id')
    return None if value is None else int(value)


def review_replay(replay):
    decisions = replay['decisions']
    counts = Counter(d['selected_action']['type'] for d in decisions)
    eligible, requested, advertised, static_advertised = set(), set(), set(), set()
    unknown_type_advertised = set()
    for decision in decisions:
        observed_own = own(decision['observation'])
        for candidate in decision.get('candidates', []):
            action = candidate['action']
            if action['type'] == 'move' and 'unit' in action:
                uid = str(action['unit'])
                advertised.add(uid)
                typeid = str(observed_own.get(uid, {}).get('typeid', ''))
                if typeid in ('1', '2', '3'):
                    eligible.add(uid)
                elif typeid == '4':
                    static_advertised.add(uid)
                elif not typeid:
                    unknown_type_advertised.add(uid)
        action = decision['selected_action']
        if action['type'] == 'move' and 'unit' in action:
            requested.add(str(action['unit']))
    complete = 'observed_steps' in replay
    frames = [{'after_step': 0, 'observation': replay['initial_observation']}]
    if complete:
        frames += replay['observed_steps']
    else:
        frames += [{'after_step': d['step'], 'observation': d['observation']} for d in decisions]
        frames += [{'after_step': replay['ending']['steps'], 'observation': replay['final_observation']}]
    paths, cumulative, revisits, gaps = {}, Counter(), Counter(), Counter()
    previous = {}
    previous_units = {}
    static_movement_violations = []
    previous_score = None
    transport_events = []
    unchanged_move_requests = 0
    transport_score_deltas = {'load': [], 'unload': []}
    for frame in frames:
        observation = frame['observation']
        units = own(observation)
        now = {uid: position(u) for uid, u in units.items() if position(u) is not None}
        for uid, pos in now.items():
            path = paths.setdefault(uid, [])
            if path and uid not in previous:
                gaps[uid] += 1
            if not path or uid not in previous or previous[uid] != pos:
                if uid in previous:
                    cumulative[uid] += distance(previous[uid], pos)
                    if (str(units[uid].get('typeid')) == '4'
                            or str(previous_units.get(uid, {}).get('typeid')) == '4'):
                        static_movement_violations.append({
                            'unit': uid, 'after_step': frame['after_step'],
                            'from_hex': previous[uid], 'to_hex': pos,
                            'violation': 'static_type4_observed_displacement'})
                    if any(p['hex'] == pos for p in path):
                        revisits[uid] += 1
                row, col = split_mapid(pos)
                path.append({'step': frame['after_step'], 'time': observation.get('time'),
                             'hex': pos, 'row': row, 'col': col,
                             'connected_to_previous': uid in previous})
        score = observation.get('score', {}).get('own')
        index = frame.get('learner_decision_index')
        if complete and frame.get('actor_is_learner') and index is not None:
            decision = decisions[index]
            action = decision['selected_action']
            kind = action['type']
            uid = str(action.get('unit'))
            if kind == 'move' and uid in previous and uid in now and previous[uid] == now[uid]:
                unchanged_move_requests += 1
            if kind in ('load', 'unload'):
                before = own(decision['observation'])
                transitions = []
                for unit_id in before.keys() & units.keys():
                    for field in ('loaded_into', 'carrying'):
                        if field in before[unit_id] and field in units[unit_id] and before[unit_id][field] != units[unit_id][field]:
                            transitions.append({'unit': unit_id, 'field': field,
                                                'before': before[unit_id][field], 'after': units[unit_id][field]})
                delta = score - previous_score if score is not None and previous_score is not None else None
                if delta is not None:
                    transport_score_deltas[kind].append(delta)
                transport_events.append({'decision_index': index, 'type': kind,
                                         'observed_state_transitions': transitions,
                                         'own_public_score_delta_during_step': delta})
        previous, previous_score, previous_units = now, score, units
    unit_metrics = []
    movers = set()
    for uid, path in sorted(paths.items()):
        moved = cumulative[uid] > 0
        if moved:
            movers.add(uid)
        unit_metrics.append({'unit': uid, 'observed_displaced': moved,
                             'observed_cumulative_hex_distance_lower_bound': cumulative[uid],
                             'first_to_last_observed_hex_distance': distance(path[0]['hex'], path[-1]['hex']),
                             'return_to_previously_observed_hex_count': revisits[uid],
                             'observation_gaps': gaps[uid], 'path': path})
    max_wait = run = 0
    for decision in decisions:
        run = run + 1 if decision['selected_action']['type'] == 'wait' else 0
        max_wait = max(max_wait, run)
    return {'schema_version': 2, 'source_checkpoint_sha256': replay.get('source_checkpoint_sha256'),
            'snapshot_sha256': replay.get('snapshot_sha256'), 'seed': replay.get('seed'),
            'learner_side': replay.get('learner_side'), 'ending': replay.get('ending'),
            'observation_sampling': 'every_episode_step' if complete else 'learner_decisions_and_end_only',
            'learner_decisions': len(decisions), 'requested_action_counts': dict(sorted(counts.items())),
            'move_requests': counts['move'], 'unique_requested_movers': len(requested),
            'total_observed_cumulative_hex_distance_lower_bound': sum(cumulative.values()),
            'sum_first_to_last_observed_hex_distance': sum(u['first_to_last_observed_hex_distance'] for u in unit_metrics),
            'return_to_previously_observed_hex_count': sum(revisits.values()),
            'unique_observed_displaced_units': len(movers),
            'unique_units_with_advertised_move_candidate': len(advertised),
            'unique_mobile_units_with_advertised_move_candidate': len(eligible),
            'eligible_mobile_typeids': [1, 2, 3],
            'advertised_static_type4_units': sorted(static_advertised),
            'advertised_units_missing_observed_typeid': sorted(unknown_type_advertised),
            'static_movement_violations': static_movement_violations,
            'has_static_movement_violation': bool(static_movement_violations),
            'observed_displaced_eligible_units': len(movers & eligible),
            'observed_displaced_eligible_fraction': len(movers & eligible) / len(eligible) if eligible else None,
            'move_requests_without_immediate_observed_displacement': unchanged_move_requests if complete else None,
            'max_consecutive_learner_wait_decisions': max_wait,
            'invalid_actions': None, 'proven_noop_actions': None,
            'transport_events': transport_events,
            'transport_public_score_changes': {
                kind: {'requests': counts[kind], 'observed_score_intervals': len(values),
                       'sum_signed_delta': sum(values) if values else None,
                       'sum_positive_delta': sum(max(0, v) for v in values) if values else None}
                for kind, values in transport_score_deltas.items()},
            'units': unit_metrics,
            'limitations': [
                'Only learner-seat observations are used. No opponent actions or private state are inspected.',
                'Displacement includes any observed own-unit relocation, including carriage; it is not a count of successful move orders.',
                'Cumulative distance is a lower bound between observed samples; intermediate path and within-step cycles may be missed.',
                'Eligibility is the union of advertised movers whose own observation at that decision identifies mobile type 1, 2, or 3. Raw engine-advertised counts remain separate; missing types are excluded.',
                'Observed static type 4 displacement violates the mobility contract. Historical outcomes and distances are retained as observed, not corrected into valid games.',
                'A return to a previously observed hex flags a possible cycle, not proof of irrational tactics.',
                'No-displacement requests may still consume time or affect other state; invalid and true noop counts are unavailable.',
                'Load/unload score deltas are contemporaneous public changes, not causal score gains.',
                'Legacy replay sampling can miss changes between learner decisions; unavailable step diagnostics are null.']}


def write_html(review, stream):
    esc = html.escape
    paths = review['units']
    points = [(p['col'] + (p['row'] % 2) / 2, p['row'] * 0.8660254)
              for u in paths for p in u['path']]
    svg = ''
    if points:
        minx, maxx = min(x for x, y in points), max(x for x, y in points)
        miny, maxy = min(y for x, y in points), max(y for x, y in points)
        chunks = [f'<svg role="img" aria-label="Observed own-unit paths in odd-r hex coordinates" viewBox="{minx-1} {miny-1} {maxx-minx+2} {maxy-miny+2}">']
        for i, unit in enumerate(paths):
            color = f'hsl({i*137.5%360:.1f},65%,40%)'
            path = unit['path']
            for j, p in enumerate(path):
                x, y = p['col'] + (p['row'] % 2) / 2, p['row'] * 0.8660254
                if j and p['connected_to_previous']:
                    prev = path[j-1]
                    px, py = prev['col'] + (prev['row'] % 2) / 2, prev['row'] * 0.8660254
                    chunks.append(f'<line x1="{px}" y1="{py}" x2="{x}" y2="{y}" stroke="{color}" stroke-width="0.08"/>')
                chunks.append(f'<circle cx="{x}" cy="{y}" r="0.12" fill="{color}"><title>{esc(unit["unit"])}: hex {p["hex"]}, step {p["step"]}</title></circle>')
        svg = ''.join(chunks) + '</svg>'
    rows = ''.join(f'<tr><td>{esc(u["unit"])}</td><td>{u["observed_cumulative_hex_distance_lower_bound"]}</td><td>{u["first_to_last_observed_hex_distance"]}</td><td>{u["return_to_previously_observed_hex_count"]}</td></tr>' for u in paths)
    stream.write('<!doctype html><html lang="en"><meta charset="utf-8"><title>Trajectory review</title><style>body{font:16px system-ui;max-width:1000px;margin:2em auto;padding:1em;color:#202a35}svg{width:100%;height:420px;background:#f4f6f8}td,th{text-align:left;padding:.4em 1em;border-bottom:1px solid #ddd}li{margin:.5em 0}</style><h1>Observed trajectory review</h1>')
    stream.write(f'<p>Seed {esc(str(review["seed"]))}; learner {esc(str(review["learner_side"]))}; {review["observation_sampling"]}.</p><p>{review["move_requests"]} requested moves; {review["unique_observed_displaced_units"]} own units actually observed displaced; {review["observed_displaced_eligible_units"]} / {review["unique_mobile_units_with_advertised_move_candidate"]} eligible mobile units observed displaced. Engine advertised {review["unique_units_with_advertised_move_candidate"]} units in total.</p>')
    if review['has_static_movement_violation']:
        stream.write('<p><strong>Invalid mobility evidence: static type 4 displacement observed. This historical trajectory is not valid evidence of playing strength.</strong></p>')
    stream.write('<p>Action requests: ' + esc(json.dumps(review['requested_action_counts'])) + f'. Longest learner wait run: {review["max_consecutive_learner_wait_decisions"]}.</p>')
    stream.write('<h2>Own-unit observed paths</h2><p>Actual odd-r hex coordinates. Lines join observed samples; they are not claimed engine routes. Hover over points for unit, hex and step.</p>' + svg)
    stream.write('<table><tr><th>Unit</th><th>Cumulative distance ≥</th><th>First→last distance</th><th>Returns</th></tr>' + rows + '</table>')
    stream.write('<h2>Load / unload</h2><pre>' + esc(json.dumps(review['transport_public_score_changes'], indent=2)) + '</pre>')
    stream.write('<h2>Interpretation limits</h2><ul>' + ''.join('<li>' + esc(s) + '</li>' for s in review['limitations']) + '</ul></html>')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--html', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.report.resolve() == args.html.resolve() or any(p.exists() for p in (args.report, args.html)):
        parser.error('choose distinct new output paths; existing files are never overwritten')
    review = review_replay(json.loads(args.replay.read_text(encoding='utf-8')))
    with args.report.open('x', encoding='utf-8') as stream:
        json.dump(review, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    with args.html.open('x', encoding='utf-8') as stream:
        write_html(review, stream)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
