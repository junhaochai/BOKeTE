"""
Unit tests for bokete training machinery (Trainer, EarlyStopping, Checkpoint).
"""

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bokete.metrics import training_report
from bokete.training import (
    Checkpoint,
    EarlyStopping,
    Trainer,
)


class TestTraining(unittest.TestCase):

    def test_early_stopping(self):
        stopper = EarlyStopping(patience=2, min_delta=0.01)
        self.assertFalse(stopper.step(1.0))
        self.assertFalse(stopper.step(0.99))
        self.assertTrue(stopper.step(0.99))

    def test_early_stopping_none_patience(self):
        stopper = EarlyStopping(patience=None)
        self.assertFalse(stopper.step(1.0))
        self.assertFalse(stopper.step(0.5))
        self.assertFalse(stopper.step(2.0))

    def test_early_stopping_on_epoch_end_sets_should_stop(self):
        stopper = EarlyStopping(patience=1)
        stopper.on_epoch_end(None, epoch=1, val_loss=1.0)
        
        class DummyTrainer:
            should_stop = False
            
        dummy_trainer = DummyTrainer()
        stopper.on_epoch_end(dummy_trainer, epoch=2, val_loss=1.0)
        self.assertTrue(dummy_trainer.should_stop)

    def test_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt = Checkpoint(directory=tmp_dir)
            model = nn.Linear(2, 2)
            ckpt.update(model, epoch=1, val_loss=0.5)

            last_path = Path(tmp_dir) / 'last.pt'
            best_path = Path(tmp_dir) / 'best.pt'

            self.assertTrue(last_path.exists())
            self.assertTrue(best_path.exists())

    def test_checkpoint_disabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            target_dir = Path(tmp_dir) / 'checkpoints'
            ckpt = Checkpoint(directory=target_dir, enabled=False)
            model = nn.Linear(2, 2)
            ckpt.update(model, epoch=1, val_loss=0.5)

            self.assertFalse(target_dir.exists())

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

    def test_trainer_timing_and_device_logging(self):
        x = torch.randn(20, 4)
        y = torch.randint(0, 2, (20,))
        loader = DataLoader(TensorDataset(x, y), batch_size=5)

        model = nn.Linear(4, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        trainer = Trainer(model, criterion, optimizer)
        metrics = trainer.fit(loader, loader, epochs=2, progress=False)

        self.assertIsNotNone(metrics.start_time)
        self.assertIsNotNone(metrics.end_time)
        self.assertIsNotNone(metrics.duration_seconds)
        self.assertIsNotNone(metrics.device_name)
        self.assertGreater(metrics.duration_seconds, 0)

        report = training_report(metrics)
        self.assertIn("start_time", report)
        self.assertIn("end_time", report)
        self.assertIn("duration_seconds", report)
        self.assertIn("duration_formatted", report)
        self.assertIn("device_name", report)


if __name__ == "__main__":
    unittest.main()
