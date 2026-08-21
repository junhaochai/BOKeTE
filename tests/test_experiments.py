"""
Unit tests for bokete experiment orchestration (run_experiments, run_trials, create_trial_runner).
"""

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bokete.experiments import create_trial_runner, run_experiments


class TestExperiments(unittest.TestCase):

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
            res = runner(1, config, output_dir=tmpdir)
            self.assertIn("extra_metrics", res)
            self.assertEqual(res["extra_metrics"]["test_accuracy"], 0.95)

    def test_run_experiments(self):
        base_config = {"experiment_name": "exp_sweep", "training": {"lr": 0.01, "epochs": 1}}
        param_grid = {"training.lr": [0.01, 0.001]}

        def dummy_run_fn(config):
            return {"train_loss": [0.5], "val_loss": [0.6]}

        results = run_experiments(base_config, param_grid, dummy_run_fn)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["parameters"]["training.lr"], 0.01)
        self.assertEqual(results[1]["parameters"]["training.lr"], 0.001)

    def test_run_experiments_nested_list(self):
        base_config = {"experiment_name": "exp_sweep", "model": {"layer_sizes": [128, 64]}}
        param_grid = {"model.layer_sizes": [[256, 128], [512, 256, 128]]}

        def dummy_run_fn(config):
            return {"train_loss": [0.5]}

        results = run_experiments(base_config, param_grid, dummy_run_fn)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["parameters"]["model.layer_sizes"], [256, 128])
        self.assertEqual(results[1]["parameters"]["model.layer_sizes"], [512, 256, 128])

    def test_run_trials(self):
        config = {"experiment_name": "trial_test", "training": {"epochs": 1}}

        def dummy_trial_fn(trial_num, cfg):
            return {"train_loss": [0.5], "val_loss": [0.6]}

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_list = run_experiments(config, num_trials=2, run_fn=dummy_trial_fn, output_dir=tmpdir)
            self.assertEqual(len(metrics_list), 2)
            self.assertTrue((Path(tmpdir) / "summary-report.md").exists())


if __name__ == "__main__":
    unittest.main()
