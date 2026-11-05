"""
Unit tests for bokete training machinery (Trainer, EarlyStopping, Checkpoint).
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bokete.metrics import training_report
from bokete.training import (
    Checkpoint,
    EarlyStopping,
    Trainer,
    evaluate,
)


class TestTraining(unittest.TestCase):

    def test_accuracy_selects_checkpoint_and_stops_on_plateau(self):
        model = nn.Linear(1, 2, bias=False)
        trainer = Trainer(model, nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr=0.1), device="cpu")
        scores = [
            {"val_loss": 0.1, "val_accuracy": 0.6},
            {"val_loss": 0.5, "val_accuracy": 0.9},
            {"val_loss": 0.4, "val_accuracy": 0.9},
            {"val_loss": 0.3, "val_accuracy": 0.8},
        ]
        epoch = 0

        def train_epoch(*args, **kwargs):
            nonlocal epoch
            epoch += 1
            with torch.no_grad():
                model.weight.fill_(epoch)
            return 0.2

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Checkpoint(tmpdir)
            stopper = EarlyStopping(patience=2)
            with patch.object(trainer, "_train_epoch", side_effect=train_epoch):
                with patch.object(trainer, "evaluate", side_effect=scores) as validation:
                    history = trainer.fit([], [], epochs=10, checkpoint=checkpoint,
                                          early_stopping=stopper, monitor="val_accuracy", progress=False)
            saved = torch.load(Path(tmpdir) / "best.pt", weights_only=True)
            last = torch.load(Path(tmpdir) / "last.pt", weights_only=True)
        self.assertEqual(validation.call_count, 4)
        self.assertEqual(history.best_epoch, 2)
        self.assertEqual(history.best_score, 0.9)
        self.assertEqual(history.best_val_loss, 0.1)
        self.assertTrue(history.stopped_early)
        self.assertEqual(stopper.best_score, 0.9)
        self.assertEqual(saved["epoch"], 2)
        self.assertEqual(saved["monitor"], "val_accuracy")
        self.assertEqual(saved["mode"], "max")
        self.assertEqual(saved["score"], 0.9)
        self.assertTrue(torch.equal(saved["model_state_dict"]["weight"], torch.full_like(model.weight, 2)))
        self.assertEqual(last["epoch"], 4)
        self.assertEqual(last["best_score"], 0.9)
        summary = training_report(history)
        self.assertEqual(summary["best_epoch"], 2)
        self.assertEqual(summary["selected_val_loss"], 0.5)
        self.assertEqual(summary["final_val_accuracy"], 0.8)

    def test_accuracy_is_sample_weighted_in_one_pass(self):
        class CountingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, inputs):
                self.calls += 1
                return inputs

        model = CountingModel()
        inputs = torch.tensor([[2., 0.], [0., 2.], [2., 0.]])
        targets = torch.tensor([0, 1, 1])
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
        result = evaluate(model, loader, nn.CrossEntropyLoss(), device="cpu", collect_accuracy=True)
        self.assertAlmostEqual(result["val_accuracy"], 2 / 3)
        self.assertEqual(model.calls, 2)
        self.assertIn("val_loss", result)

    def test_invalid_accuracy_inputs_and_empty_loader(self):
        with self.assertRaisesRegex(ValueError, "class-index"):
            evaluate(nn.Identity(), [(torch.ones(2, 1), torch.ones(2, 1))],
                     nn.MSELoss(), device="cpu", collect_accuracy=True)
        with self.assertRaisesRegex(ValueError, "empty loader"):
            evaluate(nn.Identity(), [], nn.CrossEntropyLoss(), device="cpu", collect_accuracy=True)

    def test_accuracy_excludes_ignored_targets(self):
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        targets = torch.tensor([0, 1])
        for logits, expected in (([[4., 0.], [4., 0.]], 0.0), ([[0., 4.], [0., 4.]], 1.0)):
            with self.subTest(expected=expected):
                result = evaluate(nn.Identity(), [(torch.tensor(logits), targets)], criterion,
                                  device="cpu", collect_accuracy=True)
                self.assertEqual(result["val_accuracy"], expected)

    def test_ignored_only_batches_do_not_poison_validation_loss(self):
        ignored = (torch.tensor([[4., 0.]]), torch.tensor([-100]))
        valid = (torch.tensor([[0., 4.]]), torch.tensor([1]))
        criterion = nn.CrossEntropyLoss()
        result = evaluate(nn.Identity(), [ignored, valid], criterion, device="cpu", collect_accuracy=True)
        self.assertEqual(result["val_accuracy"], 1.0)
        self.assertAlmostEqual(result["val_loss"], criterion(*valid).item())
        with self.assertRaisesRegex(ValueError, "no valid targets"):
            evaluate(nn.Identity(), [ignored], criterion, device="cpu", collect_accuracy=True)

    def test_invalid_monitor_rejected_before_training(self):
        model = nn.Linear(1, 1)
        trainer = Trainer(model, nn.MSELoss(), torch.optim.SGD(model.parameters(), lr=0.1), device="cpu")
        with self.assertRaisesRegex(ValueError, "monitor"):
            trainer.fit([], [], epochs=1, monitor="test_accuracy")

    def test_accuracy_min_delta_and_nonfinite_scores(self):
        stopper = EarlyStopping(patience=2, min_delta=0.01, monitor="val_accuracy")
        self.assertFalse(stopper.step(0.5))
        self.assertFalse(stopper.step(0.505))
        self.assertTrue(stopper.step(0.505))
        with self.assertRaisesRegex(ValueError, "finite"):
            stopper.step(float("nan"))

    def test_default_loss_monitor_uses_unrounded_scores(self):
        model = nn.Linear(1, 1)
        trainer = Trainer(model, nn.MSELoss(), torch.optim.SGD(model.parameters(), lr=0.1), device="cpu")
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Checkpoint(tmpdir)
            with patch.object(trainer, "_train_epoch", return_value=0.2):
                with patch.object(trainer, "evaluate", side_effect=[0.50004, 0.50003]):
                    history = trainer.fit([], [], epochs=2, checkpoint=checkpoint, progress=False)
            saved = torch.load(Path(tmpdir) / "best.pt", weights_only=True)
        self.assertEqual(history.best_epoch, 2)
        self.assertEqual(training_report(history)["best_epoch"], 2)
        self.assertEqual(saved["epoch"], 2)
        self.assertEqual(saved["mode"], "min")
        self.assertEqual(saved["score"], 0.50003)

    def test_plateau_scheduler_follows_accuracy_and_checks_direction(self):
        model = nn.Linear(1, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = Trainer(model, nn.CrossEntropyLoss(), optimizer, device="cpu")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=0, factor=0.5)
        with patch.object(trainer, "_train_epoch", return_value=0.2):
            with patch.object(trainer, "evaluate", side_effect=[
                {"val_loss": 0.5, "val_accuracy": 0.9},
                {"val_loss": 0.1, "val_accuracy": 0.8},
            ]):
                trainer.fit([], [], epochs=2, monitor="val_accuracy", scheduler=scheduler, progress=False)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.05)
        wrong_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
        with self.assertRaisesRegex(ValueError, "mode must match"):
            trainer.fit([], [], epochs=1, monitor="val_accuracy", scheduler=wrong_scheduler)

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
