"""
Unit tests for bokete experiment orchestration (run_experiments, run_trials, create_trial_runner).
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bokete.experiments import create_trial_runner, run_experiments
from bokete.metrics import TrainingMetrics


class TestExperiments(unittest.TestCase):

    def test_sweep_monitor_override_completes_final_report(self):
        config = {"experiment_name": "monitor_override", "training": {"monitor": "val_loss"},
                  "param_grid": {"training.monitor": ["val_accuracy"]}}

        def runner(trial_num, run_config, **kwargs):
            return {"train_loss": [0.3, 0.2], "val_loss": [0.1, 0.5],
                    "val_accuracy": [0.6, 0.9], "monitor": run_config['training']['monitor']}

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_experiments(config, run_fn=runner, output_dir=tmpdir)
            report = (Path(tmpdir) / "experiment-report.md").read_text(encoding="utf-8")
        self.assertEqual(len(result), 1)
        self.assertIn("**Selection Metric:** `val_accuracy` (maximise)", report)

    def test_trial_runner_passes_accuracy_monitor(self):
        def loaders(config):
            loader = DataLoader(TensorDataset(torch.randn(6, 2), torch.tensor([0, 1, 0, 1, 0, 1])), batch_size=4)
            return loader, loader

        runner = create_trial_runner(lambda config: nn.Linear(2, 2), loaders)
        config = {"experiment_name": "accuracy_runner", "training": {"epochs": 1, "monitor": "val_accuracy"}}
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner(1, config, output_dir=tmpdir)
            checkpoint = torch.load(Path(tmpdir) / "trial_1" / "checkpoints" / "best.pt", weights_only=True)
            report = (Path(tmpdir) / "trial_1" / "report.md").read_text(encoding="utf-8")
        self.assertEqual(result["monitor"], "val_accuracy")
        self.assertEqual(len(result["val_accuracy"]), 1)
        self.assertEqual(checkpoint["score"], result["val_accuracy"][0])
        self.assertEqual(checkpoint["mode"], "max")
        self.assertIn("Validation Accuracy", report)

    def test_custom_runner_logs_configured_monitor(self):
        config = {"experiment_name": "custom_accuracy", "training": {"monitor": "val_accuracy"}}

        def runner(*args, **kwargs):
            return {"train_loss": [0.2, 0.1], "val_loss": [0.1, 0.5], "val_accuracy": [0.6, 0.9]}

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertLogs("bokete.experiments", level="INFO") as logs:
                run_experiments(config, run_fn=runner, output_dir=tmpdir)
            report = (Path(tmpdir) / "config-report.md").read_text(encoding="utf-8")
        self.assertTrue(any("Best Validation Accuracy:" in entry for entry in logs.output))
        self.assertIn("Selected Validation Accuracy", report)

    def test_example_run(self):
        import runpy
        import sys
        project_root = str(Path(__file__).resolve().parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        orig_argv = sys.argv
        sys.argv = ["main.py", "--config", "example/configs/example.yaml"]
        try:
            res = runpy.run_path("example/main.py")
            self.assertIsNotNone(res)
        finally:
            sys.argv = orig_argv

    def test_create_trial_runner(self):
        def model_factory(cfg):
            return nn.Linear(5, 1)

        def loader_factory(cfg):
            x = torch.randn(20, 5)
            y = torch.randn(20, 1)
            ds = TensorDataset(x, y)
            loader = DataLoader(ds, batch_size=5)
            return loader, loader

        runner = create_trial_runner(model_factory, loader_factory)
        self.assertIsNotNone(runner)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "experiment_name": "test_runner",
                "seed": 42,
                "training": {"epochs": 2, "lr": 0.01, "loss": "MSELoss", "optimizer": "Adam"},
            }
            metrics = runner(1, config, output_dir=tmpdir)
            self.assertIn("train_loss", metrics)
            self.assertIn("val_loss", metrics)
            self.assertTrue((Path(tmpdir) / "trial_1" / "report.md").exists())

    def test_create_trial_runner_test_fn(self):
        def model_factory(cfg):
            return nn.Linear(5, 1)

        def loader_factory(cfg):
            x, y = torch.randn(20, 5), torch.randn(20, 1)
            ds = TensorDataset(x, y)
            loader = DataLoader(ds, batch_size=5)
            return loader, loader, loader

        def test_fn_with_loader(model, test_loader, cfg):
            return {"test_accuracy": 0.95}

        runner = create_trial_runner(model_factory, loader_factory, test_fn=test_fn_with_loader)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "experiment_name": "test_fn_test",
                "training": {"epochs": 1, "lr": 0.01, "loss": "MSELoss", "optimizer": "Adam"},
            }
            with patch("bokete.experiments.torch.load", wraps=torch.load) as load_checkpoint:
                res = runner(1, config, output_dir=tmpdir)
            self.assertEqual(load_checkpoint.call_args.kwargs["map_location"], "cpu")
            self.assertIn("extra_metrics", res)
            self.assertEqual(res["extra_metrics"]["Best test_accuracy"], 0.95)
            self.assertEqual(res["extra_metrics"]["Final test_accuracy"], 0.95)

    def test_disabled_checkpoints_only_report_final_metrics(self):
        history = TrainingMetrics(
            train_loss=[0.2, 0.1], val_loss=[0.1, 0.5], best_epoch=1, best_val_loss=0.1,
        )
        runner = create_trial_runner(
            lambda cfg: nn.Linear(1, 1),
            lambda cfg: ([], []),
            test_fn=lambda model, cfg: {"accuracy": 0.4},
        )
        config = {"experiment_name": "no_checkpoint", "training": {"save_checkpoints": False}}
        for stale_checkpoint in (False, True):
            with self.subTest(stale_checkpoint=stale_checkpoint), tempfile.TemporaryDirectory() as tmpdir:
                if stale_checkpoint:
                    checkpoint_dir = Path(tmpdir) / "trial_1" / "checkpoints"
                    checkpoint_dir.mkdir(parents=True)
                    # Invalid content must never be read when this run saves no checkpoint.
                    (checkpoint_dir / "best.pt").write_bytes(b"previous run")
                with patch("bokete.experiments.training.Trainer.fit", return_value=history):
                    with patch("bokete.experiments.reporting.trial_report"):
                        with patch("bokete.experiments.torch.load") as load_checkpoint:
                            result = runner(1, config, output_dir=tmpdir)
                load_checkpoint.assert_not_called()
                self.assertEqual(result["extra_metrics"], {"Final accuracy": 0.4})

    def test_run_experiments(self):
        base_config = {"experiment_name": "exp_sweep", "training": {"lr": 0.01, "epochs": 1}}
        param_grid = {"training.lr": [0.01, 0.001]}

        def dummy_run_fn(trial_num, config, **kwargs):
            return {"train_loss": [0.5], "val_loss": [0.6]}

        results = run_experiments(base_config, param_grid, dummy_run_fn)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["parameters"]["training.lr"], 0.01)
        self.assertEqual(results[1]["parameters"]["training.lr"], 0.001)

    def test_run_experiments_nested_list(self):
        base_config = {"experiment_name": "exp_sweep", "model": {"layer_sizes": [128, 64]}}
        param_grid = {"model.layer_sizes": [[256, 128], [512, 256, 128]]}

        def dummy_run_fn(trial_num, config, **kwargs):
            return {"train_loss": [0.5]}

        results = run_experiments(base_config, param_grid, dummy_run_fn)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["parameters"]["model.layer_sizes"], [256, 128])
        self.assertEqual(results[1]["parameters"]["model.layer_sizes"], [512, 256, 128])

    def test_run_trials(self):
        config = {"experiment_name": "trial_test", "training": {"epochs": 1}}

        def dummy_trial_fn(trial_num, cfg, **kwargs):
            return {"train_loss": [0.5], "val_loss": [0.6]}

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_list = run_experiments(config, num_trials=2, run_fn=dummy_trial_fn, output_dir=tmpdir)
            self.assertEqual(len(metrics_list), 2)
            self.assertTrue((Path(tmpdir) / "config-report.md").exists())

    def test_keyboard_interrupt_graceful_exit(self):
        config = {"experiment_name": "interrupt_test", "training": {"epochs": 1}}

        def interrupting_loader_factory(cfg):
            raise KeyboardInterrupt()

        def dummy_model_factory(cfg):
            return nn.Linear(5, 1)

        runner = create_trial_runner(dummy_model_factory, interrupting_loader_factory)
        with tempfile.TemporaryDirectory() as tmpdir:
            res = run_experiments(config, num_trials=1, run_fn=runner, output_dir=tmpdir)
            self.assertEqual(res, [])
            self.assertFalse(Path(tmpdir).exists())


if __name__ == "__main__":
    unittest.main()
