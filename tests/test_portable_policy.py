"""Run the source-only inference example independently of repository imports."""
from pathlib import Path
import subprocess
import sys
import unittest


class PortablePolicyTests(unittest.TestCase):
    def test_synthetic_roundtrip_and_facade(self):
        example = Path(__file__).resolve().parents[1] / 'examples' / 'portable_policy'
        result = subprocess.run([sys.executable, 'smoke_test.py'], cwd=example,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('smoke test passed', result.stdout)
