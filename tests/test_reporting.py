"""
Unit tests for bokete reporting functions (training_report, trial_report, config_report, experiment_report).
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch.nn as nn

from bokete.metrics import TrainingMetrics, training_report
from bokete.reporting import (
    config_report,
    experiment_report,
    trial_report,
)


class TestReporting(unittest.TestCase):

    def test_bare_history_uses_configured_monitor_in_report_and_plot(self):
        config = {"training": {"monitor": "val_accuracy"}}
        history = {"train_loss": [0.3, 0.2], "val_loss": [0.1, 0.5], "val_accuracy": [0.6, 0.9]}
        with patch("bokete.reporting.plotting.plot_loss_curves") as plot:
            report = trial_report(config, metrics=history)
        self.assertIn("**Selected Epoch:** `2` (Validation Accuracy: `0.9`)", report)
        self.assertEqual(plot.call_args.kwargs['best_epoch'], 2)
        self.assertEqual(plot.call_args.kwargs['metrics']['monitor'], 'val_accuracy')
        self.assertNotIn('monitor', history)
        aggregate = config_report(config, [history], auto_plot=False)
        self.assertIn("| Trial 1 | `2` | `0.9` |", aggregate)

    def test_trial_report_rejects_conflicting_monitor_metadata(self):
        config = {"training": {"monitor": "val_accuracy"}}
        history = {"train_loss": [0.2], "val_loss": [0.1], "monitor": "val_loss"}
        with self.assertRaisesRegex(ValueError, "different monitoring"):
            trial_report(config, metrics=history, auto_plot=False)

    def test_sweep_honours_uniform_grid_monitor_override(self):
        config = {"training": {"monitor": "val_loss"}}
        for recorded in (False, True):
            with self.subTest(recorded=recorded):
                history = {"train_loss": [0.3, 0.2], "val_loss": [0.1, 0.5], "val_accuracy": [0.6, 0.9]}
                if recorded:
                    history['monitor'] = 'val_accuracy'
                report = experiment_report(config, [{
                    "parameters": {"training.monitor": "val_accuracy"}, "metrics": [history],
                }])
                self.assertIn("**Selection Metric:** `val_accuracy` (maximise)", report)
                self.assertIn("**Best Validation Accuracy:** `0.9000", report)

    def test_accuracy_reports_and_sweep_follow_selected_metric(self):
        config = {"experiment_name": "accuracy_selection", "training": {"monitor": "val_accuracy"}}
        first = {"train_loss": [0.2, 0.1], "val_loss": [0.1, 0.5],
                 "val_accuracy": [0.6, 0.9], "monitor": "val_accuracy"}
        second = {"train_loss": [0.2, 0.1], "val_loss": [0.05, 0.04],
                  "val_accuracy": [0.7, 0.8], "monitor": "val_accuracy"}
        md = trial_report(config=config, metrics=first, auto_plot=False)
        self.assertIn("**Selected Epoch:** `2` (Validation Accuracy: `0.9`)", md)
        self.assertIn("Selected Epoch Validation Loss | `0.5`", md)
        self.assertIn("Final Validation Accuracy | `0.9`", md)
        self.assertIn("| 2 | 0.1000 | 0.5000 | 0.9000 |", md)
        aggregate = config_report(config, [second, first], auto_plot=False)
        self.assertIn("Best Trial Validation Accuracy:** `0.9000` (Trial 2)", aggregate)
        self.assertIn("| Trial 2 | `2` | `0.9` |", aggregate)
        sweep = experiment_report(config, [
            {"parameters": {"model.name": "lower_loss"}, "metrics": [second]},
            {"parameters": {"model.name": "higher_accuracy"}, "metrics": [first]},
        ])
        self.assertIn("**Configuration:** `model.name=higher_accuracy`", sweep)
        self.assertIn("| 1 | `model.name=higher_accuracy` |", sweep)
        self.assertIn("**Selection Metric:** `val_accuracy` (maximise)", sweep)

    def test_sweep_rejects_mixed_monitors(self):
        with self.assertRaisesRegex(ValueError, "different monitoring"):
            experiment_report({}, [
                {"metrics": [{"val_loss": [0.1], "monitor": "val_loss"}]},
                {"metrics": [{"val_accuracy": [0.9], "monitor": "val_accuracy"}]},
            ])

    def test_accuracy_report_requires_recorded_accuracy(self):
        with self.assertRaisesRegex(ValueError, "no accuracy history"):
            training_report({"train_loss": [0.1], "val_loss": [0.2], "monitor": "val_accuracy"})

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

    def test_trial_report(self):
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

        md = trial_report(
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

    def test_trial_report_with_metrics_object(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {'lr': 0.001, 'batch_size': 16}
            history = TrainingMetrics(
                train_loss=[0.9, 0.5, 0.2],
                val_loss=[0.95, 0.6, 0.3],
                best_epoch=3,
                best_val_loss=0.3,
            )
            graph_path = str(Path(tmp_dir) / "graph_clean.png")

            md = trial_report(
                config=config,
                metrics=history,
                graph_filename=graph_path,
                title="Simplified Training Run"
            )

            self.assertIn("# Simplified Training Run", md)
            self.assertIn("Final Train Loss", md)
            self.assertIn("`0.2`", md)
            self.assertTrue(Path(graph_path).exists())

    def test_trial_report_with_save_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {'lr': 0.001}
            history = TrainingMetrics(train_loss=[0.5], val_loss=[0.6])
            save_file = Path(tmp_dir) / "subfolder" / "report.md"

            md = trial_report(config=config, metrics=history, save_path=str(save_file))
            self.assertTrue(save_file.exists())
            self.assertEqual(save_file.read_text(encoding="utf-8"), md)

    def test_config_report(self):
        config = {"experiment_name": "quick_test", "dataset": "mixed", "lr": 0.001}
        all_metrics = [
            {"train_loss": [0.9, 0.5, 0.2], "val_loss": [0.95, 0.6, 0.3]},
            {"train_loss": [0.8, 0.4, 0.15], "val_loss": [0.9, 0.5, 0.25]},
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_file = Path(tmp_dir) / "summary.md"
            summary_md = config_report(config, all_metrics, save_path=str(save_file))
            self.assertIn("# Configuration Report: quick_test", summary_md)
            self.assertIn("- **Experiment Name:** `quick_test`", summary_md)
            self.assertIn("Mean Best Validation Loss:", summary_md)
            self.assertIn("Trial 1", summary_md)
            self.assertTrue(save_file.exists())
            self.assertEqual(save_file.read_text(encoding="utf-8"), summary_md)

    def test_trial_report_directory_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            metrics = TrainingMetrics(train_loss=[0.5, 0.3], val_loss=[0.6, 0.4])
            config = {"experiment_name": "test_exp", "lr": 0.001}

            report_md = trial_report(
                config=config,
                metrics=metrics,
                save_path=tmp_path,
                title="Test Report",
            )
            self.assertIn("Test Report", report_md)
            self.assertTrue((tmp_path / "report.md").exists())
            self.assertTrue((tmp_path / "graph.png").exists())

    def test_trial_report_with_model(self):
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        config = {"experiment_name": "model_test", "lr": 0.01}
        metrics = TrainingMetrics(train_loss=[0.5, 0.2], val_loss=[0.6, 0.3])
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_md = trial_report(
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

    def test_trial_report_timing_and_gpu(self):
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
            md = trial_report(
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

    def test_config_report_timing_and_gpu(self):
        config = {"experiment_name": "timing_test", "dataset": "mixed", "lr": 0.001}
        all_metrics = [
            {"train_loss": [0.9, 0.2], "val_loss": [0.95, 0.3]},
            {"train_loss": [0.8, 0.15], "val_loss": [0.9, 0.25]},
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_md = config_report(
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

    def test_config_report_with_model_and_filename(self):
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        config = {"experiment_name": "config_model_test", "lr": 0.001}
        all_metrics = [
            {"train_loss": [0.9, 0.2], "val_loss": [0.95, 0.3]},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_md = config_report(
                config,
                all_metrics,
                save_path=tmpdir,
                model=model,
            )
            self.assertIn("## Model Architecture", summary_md)
            self.assertIn("- **Model Class:** `Sequential`", summary_md)
            self.assertTrue((Path(tmpdir) / "config-report.md").exists())

    def test_master_experiment_report(self):
        config = {"experiment_name": "ablation_study", "dataset": "MNIST", "lr": 0.001}
        sweep_results = [
            {
                "parameters": {"model.kbl_layers": [0]},
                "metrics": [
                    {"val_loss": [0.5, 0.2], "train_loss": [0.6, 0.1], "extra_metrics": {"Test Accuracy": "0.9500"}},
                    {"val_loss": [0.4, 0.18], "train_loss": [0.5, 0.09], "extra_metrics": {"Test Accuracy": "0.9550"}},
                ],
            },
            {
                "parameters": {"model.kbl_layers": [1]},
                "metrics": [
                    {"val_loss": [0.6, 0.1], "train_loss": [0.7, 0.05], "extra_metrics": {"Test Accuracy": "0.9800"}},
                    {"val_loss": [0.5, 0.08], "train_loss": [0.6, 0.04], "extra_metrics": {"Test Accuracy": "0.9850"}},
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            md = experiment_report(
                config=config,
                sweep_results=sweep_results,
                save_path=tmpdir,
                start_time="2026-08-20 00:00:00",
                end_time="2026-08-20 00:10:00",
                duration_seconds=600.0,
            )
            self.assertIn("# Experiment Report: ablation_study", md)
            self.assertIn("Comparative Leaderboard", md)
            self.assertIn("model.kbl_layers=[1]", md)
            self.assertIn("0.0900", md)  # best val loss for combo 2: mean(0.1, 0.08) = 0.09
            self.assertIn("Mean Test Accuracy", md)
            self.assertIn("- **Best Test Accuracy:** `0.9825 ± 0.0025`", md)
            self.assertTrue((Path(tmpdir) / "experiment-report.md").exists())

    def test_master_experiment_report_no_duplicate_best_prefix(self):
        config = {"experiment_name": "prefix_test", "lr": 0.001}
        sweep_results = [
            {
                "parameters": {"training.lr": 0.001},
                "metrics": [
                    {
                        "val_loss": [0.1],
                        "train_loss": [0.1],
                        "extra_metrics": {
                            "Best accuracy": 0.9765,
                            "Best test_loss": 0.0753,
                            "Final accuracy": 0.9759,
                            "Final test_loss": 0.0778,
                        },
                    }
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            md = experiment_report(config=config, sweep_results=sweep_results, save_path=tmpdir)
            self.assertIn("## Optimal Configuration", md)
            self.assertIn("- **Configuration:** `training.lr=0.001`", md)
            self.assertIn("- **Best Accuracy:** `0.9765 ± 0.0000`", md)
            self.assertIn("- **Best Test Loss:** `0.0753 ± 0.0000`", md)
            self.assertIn("- **Final Accuracy:** `0.9759 ± 0.0000`", md)
            self.assertIn("- **Final Test Loss:** `0.0778 ± 0.0000`", md)
            self.assertNotIn("Best Best", md)
            self.assertNotIn("Best Final", md)


if __name__ == "__main__":
    unittest.main()
