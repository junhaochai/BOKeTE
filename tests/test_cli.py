import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from bokete.cli import init_project


class TestCLI(unittest.TestCase):
    def test_help_does_not_import_training_dependencies(self):
        code = """
import runpy
import sys
sys.argv = ['bokete', '--help']
try:
    runpy.run_module('bokete.cli', run_name='__main__')
finally:
    assert not {'torch', 'matplotlib', 'numpy'} & sys.modules.keys()
"""
        result = subprocess.run([sys.executable, "-B", "-c", code],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage: bokete", result.stdout)

    def test_public_exports_remain_available(self):
        import bokete

        for name in bokete.__all__:
            with self.subTest(name=name):
                self.assertIsNotNone(getattr(bokete, name))
        with self.assertRaises(AttributeError):
            getattr(bokete, "missing_public_object")

    def test_existing_target_exits_with_error(self):
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            with self.assertRaises(SystemExit) as error:
                init_project(".")
        self.assertEqual(error.exception.code, 1)
        self.assertIn("already exists", stderr.getvalue())

    def test_missing_template_exits_with_error(self):
        with patch("bokete.cli.Path.exists", return_value=False), \
                patch("bokete.cli.Path.is_dir", return_value=False):
            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                with self.assertRaises(SystemExit) as error:
                    init_project("missing-example-project")
        self.assertEqual(error.exception.code, 1)
        self.assertIn("could not be located", stderr.getvalue())
