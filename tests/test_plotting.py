"""
Unit tests for bokete metric plotting utilities.
"""

import tempfile
import unittest
from pathlib import Path

from bokete.metrics import TrainingMetrics
from bokete.plotting import plot_loss_curves, plot_multi_trial_loss_curves


class TestPlotting(unittest.TestCase):

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

    def test_plot_multi_trial_loss_curves(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            all_trial_metrics = [
                {"train_loss": [0.8, 0.4], "val_loss": [0.85, 0.45]},
                {"train_loss": [0.75, 0.35], "val_loss": [0.80, 0.40]},
            ]
            graph_path = str(Path(tmp_dir) / "multi_curves.png")
            plot_multi_trial_loss_curves(all_trial_metrics, path=graph_path)
            self.assertTrue(Path(graph_path).exists())


if __name__ == "__main__":
    unittest.main()
