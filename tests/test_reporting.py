"""
Unit tests for bokete reporting functions (training_report, experiment_report, multi_trial_report).
"""

import tempfile
import unittest
from pathlib import Path

import torch.nn as nn

from bokete.metrics import TrainingMetrics, training_report
from bokete.reporting import experiment_report, multi_trial_report


class TestReporting(unittest.TestCase):

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

    def test_experiment_report_directory_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            metrics = TrainingMetrics(train_loss=[0.5, 0.3], val_loss=[0.6, 0.4])
            config = {"experiment_name": "test_exp", "lr": 0.001}

            report_md = experiment_report(
                config=config,
                metrics=metrics,
                save_path=tmp_path,
                title="Test Report",
            )
            self.assertIn("Test Report", report_md)
            self.assertTrue((tmp_path / "report.md").exists())
            self.assertTrue((tmp_path / "graph.png").exists())

    def test_experiment_report_with_model(self):
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        config = {"experiment_name": "model_test", "lr": 0.01}
        metrics = TrainingMetrics(train_loss=[0.5, 0.2], val_loss=[0.6, 0.3])
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_md = experiment_report(
                config=config,
                metrics=metrics,
                model=model,
                save_path=tmp_dir,
                title="Model Logging Report",
            )
            self.assertIn("## Model Architecture", report_md)
            self.assertIn("- **Model Class:** `Sequential`", report_md)
            self.assertIn("Sequential", report_md)
            self.assertIn("View Model Layer Hierarchy", report_md)

    def test_experiment_report_timing_and_gpu(self):
        config = {'experiment_name': 'timing_exp', 'lr': 0.001, 'batch_size': 16}
        metrics_summary = {
            'final_train_loss': 0.2,
            'final_val_loss': 0.3,
            'mean_train_loss': 0.5,
            'mean_val_loss': 0.6,
            'best_epoch': 2,
            'start_time': '2026-08-12 12:00:00',
            'end_time': '2026-08-12 12:02:15',
            'duration_seconds': 135.0,
            'device_name': 'cuda:0 (NVIDIA GeForce RTX 4090)',
        }
        train_loss = [0.9, 0.2]
        val_loss = [0.95, 0.3]

        with tempfile.TemporaryDirectory() as tmp_dir:
            md = experiment_report(
                config=config,
                metrics_summary=metrics_summary,
                train_loss=train_loss,
                val_loss=val_loss,
                save_path=tmp_dir,
                graph_filename="graph_timing.png",
            )

            self.assertIn("- **Start Date:** `2026-08-12 12:00:00`", md)
            self.assertIn("- **End Date:** `2026-08-12 12:02:15`", md)
            self.assertIn("- **Time Taken:** `02m 15s` (`135.00s`)", md)
            self.assertIn("- **Execution Device:** `cuda:0 (NVIDIA GeForce RTX 4090)`", md)

    def test_multi_trial_report_timing_and_gpu(self):
        config = {"experiment_name": "timing_test", "dataset": "mixed", "lr": 0.001}
        all_metrics = [
            {"train_loss": [0.9, 0.2], "val_loss": [0.95, 0.3]},
            {"train_loss": [0.8, 0.15], "val_loss": [0.9, 0.25]},
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_md = multi_trial_report(
                config,
                all_metrics,
                save_path=tmp_dir,
                start_time='2026-08-12 12:00:00',
                end_time='2026-08-12 12:05:00',
                duration_seconds=300.0,
            )
            self.assertIn("- **Start Date:** `2026-08-12 12:00:00`", summary_md)
            self.assertIn("- **End Date:** `2026-08-12 12:05:00`", summary_md)
            self.assertIn("- **Total Time Taken:** `05m 00s` (`300.00s`)", summary_md)
            self.assertIn("- **Execution Device:**", summary_md)

    def test_multi_trial_report_with_model_and_filename(self):
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        config = {"experiment_name": "multi_trial_model_test", "lr": 0.001}
        all_metrics = [
            {"train_loss": [0.9, 0.2], "val_loss": [0.95, 0.3]},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_md = multi_trial_report(
                config,
                all_metrics,
                save_path=tmpdir,
                model=model,
            )
            self.assertIn("## Model Architecture", summary_md)
            self.assertIn("- **Model Class:** `Sequential`", summary_md)
            self.assertTrue((Path(tmpdir) / "summary-report.md").exists())


if __name__ == "__main__":
    unittest.main()
