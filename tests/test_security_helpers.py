import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "offline/security_helpers.py"
spec = importlib.util.spec_from_file_location("security_helpers", MODULE)
security = importlib.util.module_from_spec(spec)
spec.loader.exec_module(security)


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "site"
        self.root.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def test_normal_file(self):
        self.assertEqual(security.confined_path(self.root, "images/a.png"), str(self.root / "images/a.png"))

    def test_reject_parent_escape(self):
        for value in ("../secret", "a/../../secret", "a\\..\\..\\secret"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                security.confined_path(self.root, value)

    def test_reject_absolute_and_windows_drives(self):
        for value in ("/etc/passwd", "\\secret", "C:/secret", "C:secret", "\\\\host\\share"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                security.confined_path(self.root, value)

    def test_reject_prefix_sibling(self):
        with self.assertRaises(ValueError):
            security.confined_path(self.root, "../site-backup/secret")

    def test_reject_symlink_escape(self):
        try:
            (self.root / "link").symlink_to(Path(self.temp.name), target_is_directory=True)
        except (OSError, NotImplementedError) as exc:   # Windows 无权限/不支持创建符号链接
            self.skipTest("本机无法创建符号链接：%s" % exc)
        with self.assertRaises(ValueError):
            security.confined_path(self.root, "link/secret")

    def test_log_removes_query_and_fragment(self):
        self.assertEqual(security.safe_log_target("/resource?token=secret#password"), "/resource")

    def test_log_control_characters(self):
        self.assertNotIn("\x1b", security.safe_log_target("/path\x1b[31m"))

    def test_invalid_target(self):
        self.assertEqual(security.safe_log_target("http://[broken"), "[invalid request target]")


if __name__ == "__main__":
    unittest.main()
