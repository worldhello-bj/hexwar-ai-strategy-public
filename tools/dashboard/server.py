#!/usr/bin/env python3
"""Read-only, loopback dashboard. Never imports the engine or starts training."""
import argparse
from datetime import datetime, timezone, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import time
from urllib.parse import urlsplit, parse_qs

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parent
TZ = timezone(timedelta(hours=8))
MAX_BYTES = 32 * 1024 * 1024
MAX_FILES = 500


def timestamp(value):
    return datetime.fromtimestamp(value, TZ).isoformat(timespec='seconds')


def read_json(path):
    if path.stat().st_size > MAX_BYTES:
        raise ValueError('artifact exceeds 32 MiB read limit')
    def reject(value):
        raise ValueError('non-finite JSON number')
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=reject)


def inside(root, path):
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('artifact outside allowed run tree')
    return resolved


def compact(data):
    """Strip model weights, observations and per-game arrays from polling payload."""
    fields = ('status', 'updated_at', 'commit', 'run_id', 'episodes_attempted',
              'episodes_completed', 'episodes_target', 'updates', 'optimizer_updates',
              'elapsed_seconds', 'episodes_per_minute', 'resources', 'phase',
              'latest_report', 'latest_evaluation', 'latest_replay', 'latest_training_summary',
              'training_elapsed_seconds', 'issues', 'algorithm',
              'scoring_mode', 'scoring_version', 'observation_schema', 'training_summary',
              'evaluation_type', 'summary', 'caveat', 'source_checkpoint_sha256',
              'training_episodes_attempted', 'training_episodes_completed')
    result = {key: data[key] for key in fields if key in data}
    evaluation = data.get('evaluation')
    if isinstance(evaluation, dict):
        result['evaluation'] = {label: {key: value for key, value in item.items()
                                                if key in ('summary', 'by_seat')}
                                if isinstance(item, dict) else item
                                for label, item in evaluation.items() if label in
                                ('initial', 'trained', 'comparison', 'caveat', 'heldout_seeds')}
    for key in ('paired_comparison', 'paired'):
        if isinstance(data.get(key), dict):
            result[key] = {k: v for k, v in data[key].items() if k != 'pairs'}
    records = data.get('training')
    if isinstance(records, list):
        elapsed = sum(r.get('elapsed_seconds', 0) for r in records)
        result['batch_rollout_episodes_per_minute'] = 60 * len(records) / elapsed if elapsed > 0 else None
        result['batch_rollout_seconds'] = elapsed
        result['batch_training_records'] = len(records)
    if 'replay_type' in data:
        result.update({key: data.get(key) for key in ('replay_type', 'seed', 'learner_side', 'ending')})
        result['decisions'] = len(data.get('decisions', []))
        result['movement'] = movement(data)
        for key in ('trajectory_metrics', 'movement_metrics', 'error_metrics'):
            if key in data:
                result[key] = data[key]
    return result


def positions(observation):
    own = observation.get('own', {})
    entities = own.values() if isinstance(own, dict) else own
    return {str(e['id']): e.get('pos', e.get('hexid')) for e in entities
            if isinstance(e, dict) and 'id' in e and ('pos' in e or 'hexid' in e)}


def movement(data):
    observations = [data.get('initial_observation', {})]
    observations += [d.get('observation', {}) for d in data.get('decisions', [])]
    observations.append(data.get('final_observation', {}))
    changes = comparisons = 0
    previous = positions(observations[0])
    for observation in observations[1:]:
        current = positions(observation)
        shared = previous.keys() & current.keys()
        comparisons += len(shared)
        changes += sum(previous[k] != current[k] for k in shared)
        previous = current
    return {'observed_position_changes': changes if comparisons else None,
            'unit_observation_comparisons': comparisons,
            'caveat': 'Changes between learner observations; not action counts, path length, or proof of tactical quality. Missing or removed units are excluded.',
            'error_rate': None, 'error_caveat': 'No error labels inferred from actions or scores.'}


class Artifacts:
    def __init__(self, root):
        self.root = root.resolve()
        self.cache = {}

    def files(self):
        # Bounded three-level scan: run root, run directories, one report subdirectory.
        paths = []
        pending = [(self.root, 0)]
        scanned = 0
        limited = False
        while pending and scanned < 2000:
            directory, depth = pending.pop(0)
            if not directory.exists():
                continue
            for path in directory.iterdir():
                scanned += 1
                if scanned > 2000 or len(paths) >= MAX_FILES:
                    limited = True
                    break
                if path.is_symlink():
                    continue
                if path.is_dir() and depth < 2:
                    pending.append((path, depth + 1))
                elif path.is_file() and path.suffix == '.json' and not any(
                        word in path.name.lower() for word in ('checkpoint', 'model')):
                    paths.append(path)
            if limited:
                break
        return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True), limited or bool(pending)

    def snapshot(self):
        paths, limited = self.files()
        entries, warnings = [], []
        active = {str(path.resolve()) for path in paths}
        self.cache = {key: value for key, value in self.cache.items() if key in active}
        for path in paths:
            try:
                path = inside(self.root, path)
                stat = path.stat()
                signature = (stat.st_mtime_ns, stat.st_size)
                key = str(path)
                if key not in self.cache or self.cache[key][0] != signature:
                    data = read_json(path)
                    if not isinstance(data, dict):
                        continue
                    recognized = any(k in data for k in ('training_summary', 'evaluation', 'evaluation_type', 'replay_type')) or path.name == 'progress.json'
                    if not recognized:
                        continue
                    self.cache[key] = (signature, compact(data))
                entries.append({'path': str(path.relative_to(self.root)), 'modified_utc8': timestamp(stat.st_mtime),
                                'age_seconds': round(time.time() - stat.st_mtime, 1),
                                'bytes': stat.st_size, 'data': self.cache[key][1]})
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                warnings.append({'path': str(path.relative_to(self.root)), 'warning': str(exc)})
        live = next((entry for entry in entries if entry['path'] == 'progress.json'), None)
        return {'generated_utc8': timestamp(time.time()), 'status': (live['data'].get('status', 'unknown') if live else 'artifacts_available' if entries else 'not_started'),
                'progress': live, 'artifacts': entries, 'warnings': warnings, 'scan_limited': limited,
                'notice': 'Read-only offline artifacts. Report timestamps do not prove a process is running. Missing metrics are unavailable, never zero. Evaluation against random is not a general strength claim.'}

    def replay(self, name):
        path = inside(self.root, self.root / name)
        if path.suffix != '.json':
            raise ValueError('JSON replay required')
        data = read_json(path)
        if not isinstance(data, dict) or 'replay_type' not in data:
            raise ValueError('not a replay artifact')
        # Selected choices and observed state only; omit the potentially huge candidate arrays.
        return {**{k: data.get(k) for k in ('replay_type', 'caveat', 'ending', 'review_points', 'initial_observation', 'final_observation', 'public_board')},
                'movement': movement(data), 'decisions': [{k: v for k, v in d.items() if k != 'candidates'} for d in data.get('decisions', [])]}


def handler(artifacts):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            # Defend loopback service against browser DNS rebinding and cross-site fetch.
            host = self.headers.get('Host', '').split(':')[0]
            if host not in ('127.0.0.1', 'localhost'):
                self.send_error(403)
                return
            request = urlsplit(self.path)
            try:
                if request.path == '/api/status':
                    payload = json.dumps(artifacts.snapshot(), allow_nan=False).encode()
                    mime = 'application/json'
                elif request.path == '/api/replay':
                    payload = json.dumps(artifacts.replay(parse_qs(request.query).get('path', [''])[0]), allow_nan=False).encode()
                    mime = 'application/json'
                elif request.path in ('/', '/app.js', '/style.css'):
                    filename, mime = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'), '/style.css': ('style.css', 'text/css')}[request.path]
                    payload = (ASSETS / filename).read_bytes()
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', mime + '; charset=utf-8')
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Content-Security-Policy', "default-src 'self'; frame-ancestors 'none'; object-src 'none'")
                self.end_headers()
                self.wfile.write(payload)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                self.send_error(400, str(exc))
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs' / 'cloud-training')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(Artifacts(args.runs)))
    print(f'Read-only dashboard: http://127.0.0.1:{server.server_port} (UTC+8)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == '__main__':
    main()
