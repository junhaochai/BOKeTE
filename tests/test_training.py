"""
Unit tests for bokete training machinery (Trainer, EarlyStopping, Checkpoint).
"""

import copy
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from bokete.metrics import training_report
from bokete.training import (
    Checkpoint,
    EarlyStopping,
    Trainer,
    evaluate,
)
from bokete.utils import set_seed


class TestCheckpointResume(unittest.TestCase):

    def make_trainer(self, device="cpu", amp=False):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(0.3), nn.Linear(8, 2)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        return Trainer(model, nn.CrossEntropyLoss(), optimizer, device=device, amp=amp)

    def assert_state_equal(self, expected, actual):
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(expected, actual))
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected:
                self.assert_state_equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                self.assert_state_equal(left, right)
        else:
            self.assertEqual(expected, actual)

    def check_resume_matches_uninterrupted(self, device="cpu", amp=False):
        def components():
            trainer = self.make_trainer(device, amp)
            if device == "cpu" and callable(getattr(torch.amp, "GradScaler", None)):
                # Exercise a nonempty scaler state even on CPU-only test hosts.
                trainer.scaler = torch.amp.GradScaler("cpu", init_scale=8.0, growth_interval=2)
            scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=1, gamma=0.7)
            stopper = EarlyStopping(patience=20)
            dataset = TensorDataset(torch.arange(48).reshape(12, 4).float() / 48, torch.arange(12) % 2)
            sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(11))
            train = DataLoader(dataset, batch_size=4, sampler=sampler,
                               generator=torch.Generator().manual_seed(12))
            val = DataLoader(dataset, batch_size=6, generator=torch.Generator().manual_seed(13))

            def prepare_batch(batch, target_device):
                inputs, targets = batch
                noise = random.random() + float(np.random.normal())
                return (inputs + noise).to(target_device), targets.to(target_device)

            trainer.prepare_batch = prepare_batch
            return trainer, scheduler, stopper, train, val

        def next_random_values():
            return {
                "python": random.random(),
                "numpy": float(np.random.normal()),
                "torch": torch.rand(4),
                "cuda": [torch.rand(4, device=index) for index in range(torch.cuda.device_count())],
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(42, deterministic=device == "cuda")
            np.random.normal()  # Leave a cached Gaussian in NumPy's state.
            full, full_scheduler, full_stopper, train, val = components()
            expected = full.fit(train, val, epochs=4, scheduler=full_scheduler,
                                early_stopping=full_stopper,
                                checkpoint=Checkpoint(Path(tmpdir) / "full"), progress=False)
            expected_random = next_random_values()

            set_seed(42, deterministic=device == "cuda")
            np.random.normal()
            first, scheduler, stopper, train, val = components()
            split_dir = Path(tmpdir) / "split"
            first.fit(train, val, epochs=2, scheduler=scheduler, early_stopping=stopper,
                      checkpoint=Checkpoint(split_dir), progress=False)
            saved = torch.load(split_dir / "last.pt", weights_only=True)
            if device != "cpu" or callable(getattr(torch.amp, "GradScaler", None)):
                self.assertTrue(saved["training_state"]["scaler_state_dict"])
            self.assertEqual(saved["epoch"], 2)

            # Reconstruction consumes random numbers; resume must undo that drift.
            set_seed(99, deterministic=device == "cuda")
            resumed, scheduler, stopper, train, val = components()
            actual = resumed.fit(train, val, epochs=4, scheduler=scheduler,
                                 early_stopping=stopper, checkpoint=Checkpoint(split_dir),
                                 resume_from=split_dir / "last.pt", progress=False)
            actual_random = next_random_values()

            self.assert_state_equal(full.model.state_dict(), resumed.model.state_dict())
            self.assert_state_equal(full.optimizer.state_dict(), resumed.optimizer.state_dict())
            self.assert_state_equal(full_scheduler.state_dict(), scheduler.state_dict())
            self.assert_state_equal(full_stopper.state_dict(), stopper.state_dict())
            self.assert_state_equal(full.scaler.state_dict(), resumed.scaler.state_dict())
            self.assert_state_equal(expected_random, actual_random)
            for field in ("train_loss", "val_loss", "best_epoch", "best_score", "best_val_loss"):
                self.assertEqual(getattr(expected, field), getattr(actual, field))
            self.assertEqual(training_report(actual)["epochs_run"], 4)
            for filename in ("last.pt", "best.pt"):
                self.assert_state_equal(
                    torch.load(Path(tmpdir) / "full" / filename, weights_only=True),
                    torch.load(split_dir / filename, weights_only=True),
                )

    def test_resume_matches_uninterrupted(self):
        self.check_resume_matches_uninterrupted()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required to exercise CUDA AMP resume")
    def test_cuda_amp_resume_matches_uninterrupted(self):
        self.check_resume_matches_uninterrupted(device="cuda", amp=True)

    def test_epochs_is_total_target_and_best_checkpoint_can_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self.make_trainer()
            with patch.object(trainer, "_train_epoch", return_value=0.2), \
                    patch.object(trainer, "evaluate", return_value=0.5):
                trainer.fit([], [], epochs=10, checkpoint=Checkpoint(tmpdir), progress=False)

            for filename, target, completed in (
                ("last.pt", 20, 10), ("last.pt", 10, 10),
                ("last.pt", 5, 10), ("best.pt", 3, 1),
            ):
                with self.subTest(filename=filename, target=target):
                    resumed = self.make_trainer()
                    with patch.object(resumed, "_train_epoch", return_value=0.2) as train, \
                            patch.object(resumed, "evaluate", return_value=0.6) as validate:
                        history = resumed.fit([], [], epochs=target, progress=False,
                                              resume_from=Path(tmpdir) / filename)
                    self.assertEqual(train.call_count, max(target - completed, 0))
                    self.assertEqual(validate.call_count, train.call_count)
                    self.assertEqual(history.epochs_run, max(target, completed))
                    self.assertEqual(history.best_epoch, 1)
                    self.assertEqual(history.best_score, 0.5)
                    self.assertEqual([call.kwargs["epoch_idx"] for call in train.call_args_list],
                                     list(range(completed + 1, target + 1)))

    def test_resume_preserves_selection_patience_and_callback_order(self):
        for monitor, scores in (("val_loss", [1.0, 0.995]), ("val_accuracy", [0.5, 0.505])):
            with self.subTest(monitor=monitor), tempfile.TemporaryDirectory() as tmpdir:
                def callbacks(trainer):
                    checkpoint = Checkpoint(tmpdir)
                    stopper = EarlyStopping(patience=3, min_delta=0.01)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        trainer.optimizer, mode="min" if monitor == "val_loss" else "max",
                        patience=0, factor=0.5,
                    )
                    return checkpoint, stopper, scheduler

                def validation(score):
                    return score if monitor == "val_loss" else {"val_loss": 0.2, "val_accuracy": score}

                trainer = self.make_trainer()
                original_callbacks = callbacks(trainer)
                with patch.object(trainer, "_train_epoch", return_value=0.2), \
                        patch.object(trainer, "evaluate", side_effect=[validation(score) for score in scores]):
                    trainer.fit([], [], epochs=2, callbacks=original_callbacks, monitor=monitor, progress=False)
                saved = torch.load(Path(tmpdir) / "last.pt", weights_only=True)
                self.assertEqual(saved["training_state"]["callback_states"][0]["bad_epochs"], 1)
                self.assertEqual(saved["training_state"]["callback_states"][1]["last_epoch"], 2)
                best_before = (Path(tmpdir) / "best.pt").read_bytes()

                resumed = self.make_trainer()
                restored_callbacks = callbacks(resumed)
                restored_callbacks[1].patience = 99
                with patch.object(resumed, "_train_epoch", return_value=0.2) as train, \
                        patch.object(resumed, "evaluate", return_value=validation(scores[-1])):
                    history = resumed.fit([], [], epochs=10, callbacks=restored_callbacks,
                                          monitor=monitor, resume_from=Path(tmpdir) / "last.pt", progress=False)
                self.assertEqual(train.call_count, 2)
                self.assertEqual(history.best_epoch, 2)
                self.assertEqual(history.best_score, scores[-1])
                self.assertEqual(restored_callbacks[1].best_score, scores[0])
                self.assertEqual(restored_callbacks[1].patience, 3)
                self.assertEqual(restored_callbacks[1].bad_epochs, 3)
                self.assertEqual(restored_callbacks[2].last_epoch, 4)
                self.assertAlmostEqual(resumed.optimizer.param_groups[0]["lr"], 0.0025)
                self.assertTrue(history.stopped_early)
                self.assertEqual((Path(tmpdir) / "best.pt").read_bytes(), best_before)

                stopped = self.make_trainer()
                with patch.object(stopped, "_train_epoch") as train:
                    history = stopped.fit([], [], epochs=10, callbacks=callbacks(stopped),
                                          monitor=monitor, resume_from=Path(tmpdir) / "last.pt", progress=False)
                train.assert_not_called()
                self.assertTrue(history.stopped_early)
                self.assertEqual(history.epochs_run, 4)

    def test_resume_rejects_legacy_and_incompatible_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self.make_trainer()
            path = Path(tmpdir) / "last.pt"
            checkpoint = Checkpoint(tmpdir)
            checkpoint.update(trainer.model, epoch=1, val_loss=0.5, optimizer=trainer.optimizer)
            with self.assertRaisesRegex(ValueError, "full training state"):
                trainer.fit([], [], epochs=2, resume_from=path, progress=False)

            with patch.object(trainer, "_train_epoch", return_value=0.2), \
                    patch.object(trainer, "evaluate", return_value=0.5):
                trainer.fit([], [], epochs=1, checkpoint=checkpoint, progress=False)
            saved = torch.load(path, weights_only=True)
            cases = [
                ("version", 99, "full training state"),
                ("amp_active", True, "AMP configuration"),
                ("callback_types", ["missing.Callback"], "callback types and order"),
                ("callback_states", [{}], "callback state is incomplete"),
            ]
            for field, value, message in cases:
                with self.subTest(field=field):
                    incompatible = copy.deepcopy(saved)
                    incompatible["training_state"][field] = value
                    torch.save(incompatible, path)
                    resumed = self.make_trainer()
                    with patch.object(resumed, "_train_epoch") as train, self.assertRaisesRegex(ValueError, message):
                        resumed.fit([], [], epochs=2, resume_from=path, progress=False)
                    train.assert_not_called()

            torch.save(saved, path)
            resumed = self.make_trainer()
            with self.assertRaisesRegex(ValueError, "same monitored metric"):
                resumed.fit([], [], epochs=2, monitor="val_accuracy", resume_from=path, progress=False)
            scheduler = torch.optim.lr_scheduler.StepLR(resumed.optimizer, step_size=1)
            with self.assertRaisesRegex(ValueError, "callback types and order"):
                resumed.fit([], [], epochs=2, scheduler=scheduler, resume_from=path, progress=False)
            loader = DataLoader(TensorDataset(torch.ones(2, 4), torch.zeros(2, dtype=torch.long)),
                                generator=torch.Generator().manual_seed(1))
            with self.assertRaisesRegex(ValueError, "DataLoader generator configuration"):
                resumed.fit(loader, [], epochs=2, resume_from=path, progress=False)
            saved["training_state"]["rng_state"]["cuda"].append(torch.get_rng_state())
            torch.save(saved, path)
            with self.assertRaisesRegex(ValueError, "visible CUDA devices"):
                resumed.fit([], [], epochs=2, resume_from=path, progress=False)


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
