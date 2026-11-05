import io
import unittest
from unittest.mock import patch

from bokete.cli import init_project


class TestCLI(unittest.TestCase):
    def test_existing_target_exits_with_error(self):
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            with self.assertRaises(SystemExit) as error:
                init_project(".")
        self.assertEqual(error.exception.code, 1)
        self.assertIn("already exists", stderr.getvalue())

    def test_missing_example_exits_with_error(self):
        with patch("bokete.cli.Path.exists", return_value=False):
            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                with self.assertRaises(SystemExit) as error:
                    init_project("missing-example-project")
        self.assertEqual(error.exception.code, 1)
        self.assertIn("could not be located", stderr.getvalue())
