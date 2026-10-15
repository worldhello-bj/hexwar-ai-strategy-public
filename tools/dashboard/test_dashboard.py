"""Synthetic artifact fixtures only: these tests do not run or claim training."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer
from server import Artifacts, compact, handler, movement


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.artifacts = Artifacts(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return path

    def test_empty_does_not_claim_training(self):
        snapshot = self.artifacts.snapshot()
        self.assertEqual(snapshot['status'], 'not_started')
        self.assertEqual(snapshot['artifacts'], [])

    def test_progress_and_report(self):
        self.write('progress.json', {'status': 'training', 'episodes_completed': 4,
                                    'episodes_attempted': 5, 'episodes_target': 10,
                                    'commit': 'fixture-only', 'resources': {'rss_mb': 40}})
        self.write('fixture/training.json', {'training_summary': {'completed': 4, 'truncated': 1},
                    'training': [{'elapsed_seconds': 10}, {'elapsed_seconds': 20}],
                    'evaluation': {'trained': {'summary': {'win': 1}, 'games': ['large']}},
                    'policy': {'must_not_expose': 'weights'}})
        self.write('fixture/checkpoint.json', {'policy': 'never read'})
        result = self.artifacts.snapshot()
        self.assertEqual(result['status'], 'training')
        self.assertEqual(len(result['artifacts']), 2)
        report = next(a for a in result['artifacts'] if a['path'].endswith('training.json'))
        self.assertEqual(report['data']['batch_rollout_episodes_per_minute'], 4)
        self.assertNotIn('policy', report['data'])
        self.assertNotIn('games', report['data']['evaluation']['trained'])
        self.assertTrue(report['modified_utc8'].endswith('+08:00'))

    def test_partial_json_and_symlink(self):
        (self.root / 'progress.json').write_text('{')
        (self.root / 'linked.json').symlink_to('/etc/passwd')
        result = self.artifacts.snapshot()
        self.assertEqual(len(result['warnings']), 1)
        self.assertEqual(result['status'], 'not_started')

    def test_movement_not_action_or_error_inference(self):
        replay = {'replay_type': 'fixture', 'initial_observation': {'own': {'u': {'id': 'u', 'pos': '1'}}},
                  'decisions': [{'observation': {'own': {'u': {'id': 'u', 'pos': '2'}}}, 'candidates': ['omitted']}],
                  'final_observation': {'own': {'u': {'id': 'u', 'pos': '2'}}}}
        self.write('r/replay.json', replay)
        result = self.artifacts.replay('r/replay.json')
        self.assertEqual(result['movement']['observed_position_changes'], 1)
        self.assertIsNone(result['movement']['error_rate'])
        self.assertNotIn('candidates', result['decisions'][0])
        self.assertIsNone(movement({})['observed_position_changes'])
        with self.assertRaises(ValueError):
            self.artifacts.replay('../outside.json')

    def test_http_read_only_routes(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler(self.artifacts))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = 'http://127.0.0.1:' + str(server.server_port)
        try:
            with urlopen(url + '/api/status') as response:
                self.assertEqual(json.load(response)['status'], 'not_started')
            with urlopen(url) as response:
                self.assertIn('frame-ancestors', response.headers['Content-Security-Policy'])
            for request, code in [(Request(url, headers={'Host': 'evil.invalid'}), 403),
                                  (Request(url + '/api/status', data=b'{}'), 501),
                                  (Request(url + '/etc/passwd'), 404)]:
                with self.assertRaises(HTTPError) as exc:
                    urlopen(request)
                self.assertEqual(exc.exception.code, code)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

if __name__ == '__main__':
    unittest.main()
