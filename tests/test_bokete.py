"""
Unit tests for bokete primitives and helper functions.
"""

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bokete import (
    Checkpoint,
    EarlyStopping,
    Trainer,
    TrainingMetrics,
    determine_device,
    experiment_report,
    multi_trial_report,
    plot_loss_curves,
    set_seed,
    training_report,
)


class TestBokete(unittest.TestCase):

    def test_set_seed(self):
        set_seed(42)
        r1 = torch.randn(5)
        set_seed(42)
        r2 = torch.randn(5)
        self.assertTrue(torch.allclose(r1, r2))

    def test_determine_device(self):
        device = determine_device()
        self.assertIsInstance(device, torch.device)

    def test_early_stopping(self):
        stopper = EarlyStopping(patience=2, min_delta=0.01)
        self.assertFalse(stopper.step(1.0))
        self.assertFalse(stopper.step(0.99))  # no improvement > min_delta
        self.assertTrue(stopper.step(0.99))   # bad_epochs >= patience

    def test_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt = Checkpoint(directory=tmp_dir)
            model = nn.Linear(2, 2)
            ckpt.update(model, epoch=1, val_loss=0.5)

            last_path = Path(tmp_dir) / 'last.pt'
            best_path = Path(tmp_dir) / 'best.pt'

            self.assertTrue(last_path.exists())
            self.assertTrue(best_path.exists())

    def test_training_report(self):
        metrics_dict = {
            'train_loss': [0.9, 0.5, 0.2],
            'val_loss': [0.95, 0.6, 0.3],
            'best_epoch': 3,
        }
        summary = training_report(metrics_dict)
        self.assertEqual(summary['final_train_loss'], 0.2)
        self.assertEqual(summary['final_val_loss'], 0.3)
        self.assertEqual(summary['best_epoch'], 3)

    def test_experiment_report(self):
        config = {'lr': 0.001, 'batch_size': 16}
        metrics_summary = {
            'final_train_loss': 0.2,
            'final_val_loss': 0.3,
            'mean_train_loss': 0.5333,
            'mean_val_loss': 0.6167,
            'best_epoch': 3,
        }
        train_loss = [0.9, 0.5, 0.2]
        val_loss = [0.95, 0.6, 0.3]

        md = experiment_report(
            config=config,
            metrics_summary=metrics_summary,
            train_loss=train_loss,
            val_loss=val_loss,
            graph_filename="graph_1.png",
            extra_metrics={"Convergence Speed": 0.2}
        )

        self.assertIn("# Experiment Report", md)
        self.assertIn("Final Train Loss", md)
        self.assertIn("Convergence Speed", md)
        self.assertIn("graph_1.png", md)

    def test_experiment_report_with_metrics_object(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {'lr': 0.001, 'batch_size': 16}
            history = TrainingMetrics(
                train_loss=[0.9, 0.5, 0.2],
                val_loss=[0.95, 0.6, 0.3],
                best_epoch=3,
                best_val_loss=0.3,
            )
            graph_path = str(Path(tmp_dir) / "graph_clean.png")

            md = experiment_report(
                config=config,
                metrics=history,
                graph_filename=graph_path,
                title="Simplified Training Run"
            )

            self.assertIn("# Simplified Training Run", md)
            self.assertIn("Final Train Loss", md)
            self.assertIn("`0.2`", md)
            self.assertTrue(Path(graph_path).exists())

    def test_plot_loss_curves_with_metrics_object(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            history = TrainingMetrics(
                train_loss=[0.8, 0.4],
                val_loss=[0.85, 0.45],
                best_epoch=2,
            )
            graph_path = str(Path(tmp_dir) / "curves.png")
            plot_loss_curves(history, path=graph_path)
            self.assertTrue(Path(graph_path).exists())

    def test_experiment_report_with_save_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {'lr': 0.001}
            history = TrainingMetrics(train_loss=[0.5], val_loss=[0.6])
            save_file = Path(tmp_dir) / "subfolder" / "report.md"

            md = experiment_report(config=config, metrics=history, save_path=str(save_file))
            self.assertTrue(save_file.exists())
            self.assertEqual(save_file.read_text(encoding="utf-8"), md)

    def test_multi_trial_report(self):
        config = {"experiment_name": "quick_test", "dataset": "mixed", "lr": 0.001}
        all_metrics = [
            {"train_loss": [0.9, 0.5, 0.2], "val_loss": [0.95, 0.6, 0.3]},
            {"train_loss": [0.8, 0.4, 0.15], "val_loss": [0.9, 0.5, 0.25]},
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_file = Path(tmp_dir) / "summary.md"
            summary_md = multi_trial_report(config, all_metrics, save_path=str(save_file))
            self.assertIn("# Multi-Trial Experiment Summary: quick_test", summary_md)
            self.assertIn("- **Experiment Name:** `quick_test`", summary_md)
            self.assertIn("Mean Best Validation Loss:", summary_md)
            self.assertIn("Trial 1", summary_md)
            self.assertTrue(save_file.exists())
            self.assertEqual(save_file.read_text(encoding="utf-8"), summary_md)

    def test_trainer_fit(self):
        x = torch.randn(20, 4)
        y = torch.randint(0, 2, (20,))
        loader = DataLoader(TensorDataset(x, y), batch_size=5)

        model = nn.Linear(4, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        trainer = Trainer(model, criterion, optimizer)
        metrics = trainer.fit(loader, loader, epochs=2, progress=False)

        self.assertEqual(len(metrics.train_loss), 2)
        self.assertEqual(len(metrics.val_loss), 2)

    def test_trainer_evaluate(self):
        x = torch.randn(20, 4)
        y = torch.randint(0, 2, (20,))
        loader = DataLoader(TensorDataset(x, y), batch_size=5)

        model = nn.Linear(4, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        trainer = Trainer(model, criterion, optimizer)
        loss = trainer.evaluate(loader)
        self.assertIsInstance(loss, float)

    def test_trainer_fit_with_callbacks_list(self):
        x = torch.randn(20, 4)
        y = torch.randint(0, 2, (20,))
        loader = DataLoader(TensorDataset(x, y), batch_size=5)

        model = nn.Linear(4, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        early_stop = EarlyStopping(patience=10)

        trainer = Trainer(model, criterion, optimizer)
        metrics = trainer.fit(loader, loader, epochs=2, callbacks=[early_stop], progress=False)
        self.assertEqual(len(metrics.train_loss), 2)

    def test_trainer_train_epoch(self):
        x = torch.randn(20, 4)
        y = torch.randint(0, 2, (20,))
        loader = DataLoader(TensorDataset(x, y), batch_size=5)

        model = nn.Linear(4, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        trainer = Trainer(model, criterion, optimizer)
        loss = trainer._train_epoch(loader)
        self.assertIsInstance(loss, float)

    def test_example_run(self):
        import runpy
        import sys
        from pathlib import Path
        project_root = str(Path(__file__).resolve().parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        res = runpy.run_path("example/main.py")
        self.assertIsNotNone(res)

    def test_get_nested_key(self):
        from bokete import get_nested_key

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
        from bokete import load_config_as

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
        from bokete import parse_cli_config

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

            # Test default_config fallback when no --config flag is passed
            cfg_default = parse_cli_config(
                DummyConfig,
                default_config=config_path,
                args_list=[],
            )
            self.assertIsInstance(cfg_default, DummyConfig)

    def test_experiment_report_directory_path(self):
        import tempfile
        from bokete import experiment_report, TrainingMetrics

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            metrics = TrainingMetrics(train_loss=[0.5, 0.3], val_loss=[0.6, 0.4])
            config = {"experiment_name": "test_exp", "lr": 0.001}

            # Pass Path object pointing to a directory
            report_md = experiment_report(
                config=config,
                metrics=metrics,
                save_path=tmp_path,
                title="Test Report",
            )
            self.assertIn("Test Report", report_md)
            self.assertTrue((tmp_path / "report.md").exists())
            self.assertTrue((tmp_path / "graph.png").exists())

    def test_create_run_directory(self):
        import tempfile
        from dataclasses import dataclass
        from bokete import create_run_directory

        @dataclass
        class SampleConfig:
            experiment_name: str = "my_experiment"
            output_dir: str = ""

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SampleConfig(output_dir=tmpdir)
            run_dir = create_run_directory(cfg, attach_file_logger=False)
            self.assertTrue(run_dir.exists())
            self.assertEqual(run_dir.parent.name, "my_experiment")


if __name__ == '__main__':
    unittest.main()
