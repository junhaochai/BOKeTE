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

        self.assertIn("# Experiment Trial Report", md)
        self.assertIn("Final Train Loss", md)
        self.assertIn("Convergence Speed", md)
        self.assertIn("graph_1.png", md)

    def test_multi_trial_report(self):
        config = {"experiment_name": "quick_test", "dataset": "mixed", "lr": 0.001}
        all_metrics = [
            {"train_loss": [0.9, 0.5, 0.2], "val_loss": [0.95, 0.6, 0.3]},
            {"train_loss": [0.8, 0.4, 0.15], "val_loss": [0.9, 0.5, 0.25]},
        ]
        summary_md = multi_trial_report(config, all_metrics)
        self.assertIn("# Multi-Trial Experiment Summary: quick_test", summary_md)
        self.assertIn("- **Experiment Name:** `quick_test`", summary_md)
        self.assertIn("Mean Best Validation Loss:", summary_md)
        self.assertIn("Trial 1", summary_md)

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


if __name__ == '__main__':
    unittest.main()
