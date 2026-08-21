"""
Unit tests for bokete utility functions (seeds, hardware detection, config I/O, logging).
"""

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from unittest.mock import patch, MagicMock

from bokete.utils import (
    create_progress_bar,
    create_run_directory,
    determine_device,
    format_duration,
    format_epoch_label,
    format_tag,
    get_device_name,
    get_nested_key,
    load_config_as,
    log_experiment_start,
    log_model_info,
    log_trial_start,
    parse_cli_config,
    resolve_config_path,
    resolve_output_path,
    set_seed,
    setup_logging,
)


class TestUtils(unittest.TestCase):

    def test_set_seed(self):
        set_seed(42)
        r1 = torch.randn(5)
        set_seed(42)
        r2 = torch.randn(5)
        self.assertTrue(torch.allclose(r1, r2))

    def test_determine_device(self):
        device = determine_device()
        self.assertIsInstance(device, torch.device)

    def test_get_device_name(self):
        dev_name = get_device_name()
        self.assertIsInstance(dev_name, str)
        self.assertGreater(len(dev_name), 0)

    def test_format_duration(self):
        self.assertEqual(format_duration(None), "N/A")
        self.assertEqual(format_duration(12.34), "12.34s")
        self.assertEqual(format_duration(135.0), "02m 15s")
        self.assertEqual(format_duration(3665.0), "01h 01m 05s")

    def test_get_nested_key(self):
        cfg = {
            "experiment_name": "test_exp",
            "training": {"lr": 0.001, "empty": None},
            "missing_dict": None,
        }

        self.assertEqual(get_nested_key(cfg, "experiment_name"), "test_exp")
        self.assertEqual(get_nested_key(cfg, "training.lr"), 0.001)
        self.assertEqual(get_nested_key(cfg, "training.epochs", default=10), 10)
        self.assertEqual(get_nested_key(cfg, "training.empty", default=5), 5)
        self.assertEqual(get_nested_key(cfg, "missing_dict.lr", default=0.01), 0.01)
        self.assertEqual(get_nested_key(cfg, "nonexistent.path", default="fallback"), "fallback")

    def test_load_config_as(self):
        from dataclasses import dataclass, field

        @dataclass
        class SubConfig:
            lr: float = 1e-3
            epochs: int = 10

        @dataclass
        class RootConfig:
            experiment_name: str = "default_exp"
            sub: SubConfig = field(default_factory=SubConfig)

        config_path = Path("example/configs/example.yaml")
        if config_path.exists():
            parsed = load_config_as(config_path, RootConfig)
            self.assertIsInstance(parsed, RootConfig)
            self.assertIsInstance(parsed.sub, SubConfig)

    def test_parse_cli_config(self):
        from dataclasses import dataclass

        @dataclass
        class DummyConfig:
            experiment_name: str = "default_exp"

        config_path = Path("example/configs/example.yaml")
        if config_path.exists():
            cfg = parse_cli_config(
                DummyConfig,
                args_list=["--config", str(config_path)],
            )
            self.assertIsInstance(cfg, DummyConfig)

            cfg_default = parse_cli_config(
                DummyConfig,
                default_config=config_path,
                args_list=[],
            )
            self.assertIsInstance(cfg_default, DummyConfig)

    def test_resolve_config_path(self):
        # 1. Exact path resolution
        resolved = resolve_config_path("example/configs/example.yaml")
        self.assertEqual(resolved, Path("example/configs/example.yaml"))

        # 2. Short name / configs/ folder resolution via mocking
        with patch.object(Path, "is_file", side_effect=lambda: True):
            res_short = resolve_config_path("example")
            self.assertEqual(res_short, Path("configs/example.yaml"))

    def test_resolve_output_path(self):
        self.assertIsNone(resolve_output_path(None, "report.md"))
        self.assertEqual(resolve_output_path("custom/report.md", "report.md"), Path("custom/report.md"))
        self.assertEqual(resolve_output_path("custom_dir", "report.md"), Path("custom_dir/report.md"))

    def test_create_progress_bar(self):
        pbar = create_progress_bar(range(10), desc="Testing", disable=True)
        self.assertIsNotNone(pbar)
        pbar.close()

    def test_create_run_directory(self):
        from dataclasses import dataclass

        @dataclass
        class SampleConfig:
            experiment_name: str = "my_experiment"

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SampleConfig()
            run_dir = create_run_directory(cfg, base_dir=tmpdir, attach_file_logger=False)
            self.assertTrue(run_dir.exists())
            self.assertEqual(run_dir.parent.name, "my_experiment")

    def test_log_model_info(self):
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        info = log_model_info(model, log_layers=True)
        self.assertEqual(info["model_name"], "Sequential")
        self.assertEqual(info["total_params"], 61)
        self.assertEqual(info["trainable_params"], 61)
        self.assertIn("Linear", info["structure"])

    def test_log_trial_start_validation(self):
        log_trial_start(1, 5, "baseline_run")
        log_trial_start(1, 5, experiment_name="baseline_run")

        with self.assertRaises(ValueError):
            log_trial_start(1, 5, "")

        with self.assertRaises(ValueError):
            log_trial_start(1, 5, "   ")

        with self.assertRaises(ValueError):
            log_trial_start(1, 5, None)  # type: ignore

    def test_log_experiment_start(self):
        log_experiment_start(1, 4, "exp_sweep", params_str="training.lr=0.01, training.batch_size=32")
        with self.assertRaises(ValueError):
            log_experiment_start(1, 4, "")



    def test_setup_logging(self):
        import logging
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            setup_logging(log_file=log_path)
            
            logger = logging.getLogger("test_logger")
            logger.info("Test log entry")

            self.assertTrue(log_path.exists())
            content = log_path.read_text(encoding="utf-8")
            self.assertIn("Test log entry", content)

            root_logger = logging.getLogger()
            for h in list(root_logger.handlers):
                if isinstance(h, logging.FileHandler):
                    h.close()
                    root_logger.removeHandler(h)

    def test_design_system_helpers(self):
        epoch_str = format_epoch_label(1, 5)
        self.assertIn("Epoch", epoch_str)
        self.assertIn("1", epoch_str)
        self.assertIn("5", epoch_str)

        tag_str = format_tag("test_exp")
        self.assertIn("[test_exp]", tag_str)


if __name__ == "__main__":
    unittest.main()
