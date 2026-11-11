"""Regression coverage for the repository audit findings."""

import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import io
from pathlib import Path
import re
import shutil
import tempfile
from threading import Barrier
from typing import Optional
import unittest
from unittest.mock import patch, sentinel

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import bokete
from bokete.cli import init_project
from bokete.experiments import create_trial_runner, run_experiments
from bokete.metrics import TrainingMetrics
from bokete.reporting import config_report
from bokete.training import Checkpoint, Trainer, evaluate
from bokete.utils import create_run_directory, load_config, load_config_as, resolve_config_path


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class _TrainingConfig:
    epochs: int = 5
    patience: Optional[int] = 3


@dataclass
class _DeferredConfig:
    training: "_TrainingConfig" = field(default_factory=_TrainingConfig)


class TestTrainingRegressions(unittest.TestCase):
    def test_legacy_scaler_api_is_used_when_modern_api_is_absent(self):
        model = nn.Linear(1, 1)
        with patch.object(torch.amp, "GradScaler", None, create=True), \
                patch.object(torch.cuda.amp, "GradScaler", return_value=sentinel.scaler) as legacy:
            trainer = Trainer(model, nn.MSELoss(), torch.optim.SGD(model.parameters(), lr=0.1), device="cpu")
        self.assertIs(trainer.scaler, sentinel.scaler)
        legacy.assert_called_once_with(enabled=False)

    def test_failed_checkpoint_save_preserves_previous_file_and_score(self):
        for filename in ("last.pt", "best.pt"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as tmpdir:
                checkpoint = Checkpoint(tmpdir, save_last=filename == "last.pt", save_best=filename == "best.pt")
                model = nn.Linear(1, 1)
                checkpoint.update(model, epoch=1, val_loss=0.5)
                path = Path(tmpdir) / filename
                original = path.read_bytes()

                def fail_save(state, destination):
                    Path(destination).write_bytes(b"partial checkpoint")
                    raise OSError("disk full")

                with patch("bokete.training.torch.save", side_effect=fail_save), self.assertRaises(OSError):
                    checkpoint.update(model, epoch=2, val_loss=0.4)
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(torch.load(path, weights_only=True)["epoch"], 1)
                self.assertEqual(checkpoint.best_score, 0.5)
                self.assertEqual(list(Path(tmpdir).iterdir()), [path])

    def test_uneven_batches_preserve_dataset_loss_and_model_ranking(self):
        targets = torch.zeros(3, 1)
        for reduction, expected in (("mean", 3.0), ("sum", 9.0)):
            with self.subTest(reduction=reduction):
                loader = DataLoader(TensorDataset(torch.tensor([[0.], [0.], [3.]]), targets), batch_size=2)
                actual = evaluate(nn.Identity(), loader, nn.MSELoss(reduction=reduction), device="cpu")
                self.assertAlmostEqual(actual, expected)
        better = DataLoader(TensorDataset(torch.tensor([[0.], [0.], [3.]]), targets), batch_size=2)
        worse = DataLoader(TensorDataset(torch.full((3, 1), 2.0), targets), batch_size=2)
        self.assertLess(evaluate(nn.Identity(), better, nn.MSELoss(), device="cpu"),
                        evaluate(nn.Identity(), worse, nn.MSELoss(), device="cpu"))

    def test_training_loss_uses_the_same_epoch_aggregation(self):
        for reduction, expected in (("mean", 3.0), ("sum", 9.0)):
            with self.subTest(reduction=reduction):
                model = nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    model.weight.fill_(1)
                trainer = Trainer(model, nn.MSELoss(reduction=reduction),
                                  torch.optim.SGD(model.parameters(), lr=0), device="cpu")
                loader = DataLoader(TensorDataset(torch.tensor([[0.], [0.], [3.]]), torch.zeros(3, 1)), batch_size=2)
                self.assertEqual(trainer._train_epoch(loader, show_progress=False), expected)

    def test_weighted_class_losses_and_ignored_batches_match_full_dataset(self):
        logits = torch.tensor([[4., 0.], [4., 0.], [1., 0.], [3., 0.], [2., 0.], [0., 1.]])
        targets = torch.tensor([-100, -100, 0, 0, 1, 1])
        for loss_type in (nn.CrossEntropyLoss, nn.NLLLoss):
            inputs = logits if loss_type is nn.CrossEntropyLoss else logits.log_softmax(dim=1)
            for reduction in ("mean", "sum"):
                for accuracy in (False, True):
                    with self.subTest(loss=loss_type.__name__, reduction=reduction, accuracy=accuracy):
                        criterion = loss_type(weight=torch.tensor([0.2, 2.]), reduction=reduction)
                        loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
                        result = evaluate(nn.Identity(), loader, criterion, device="cpu", collect_accuracy=accuracy)
                        actual = result["val_loss"] if accuracy else result
                        self.assertAlmostEqual(actual, criterion(inputs, targets).item(), places=6)

    def test_probability_targets_and_label_smoothing_match_full_loss(self):
        logits = torch.tensor([[1., 0.], [3., 0.], [2., 0.]])
        for targets in (torch.tensor([0, 0, 1]), torch.tensor([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])):
            with self.subTest(target_shape=targets.shape):
                criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 2.]), label_smoothing=0.1)
                loader = DataLoader(TensorDataset(logits, targets), batch_size=2)
                self.assertAlmostEqual(evaluate(nn.Identity(), loader, criterion, device="cpu"),
                                       criterion(logits, targets).item(), places=6)

    def test_batchmean_reduction_matches_full_dataset(self):
        logits = torch.tensor([[1., 0.], [3., 0.], [2., 0.]]).log_softmax(dim=1)
        targets = torch.tensor([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
        criterion = nn.KLDivLoss(reduction="batchmean")
        loader = DataLoader(TensorDataset(logits, targets), batch_size=2)
        self.assertAlmostEqual(evaluate(nn.Identity(), loader, criterion, device="cpu"),
                               criterion(logits, targets).item(), places=6)

    def test_zero_weight_classes_count_towards_accuracy_and_smoothed_loss(self):
        logits = torch.tensor([[0., 4.], [0., 4.]])
        targets = torch.tensor([0, 1])
        for smoothing in (0.0, 0.2):
            with self.subTest(smoothing=smoothing):
                criterion = nn.CrossEntropyLoss(weight=torch.tensor([0., 1.]), label_smoothing=smoothing)
                loader = DataLoader(TensorDataset(logits, targets), batch_size=1)
                result = evaluate(nn.Identity(), loader, criterion, device="cpu", collect_accuracy=True)
                self.assertEqual(result["val_accuracy"], 0.5)
                self.assertAlmostEqual(result["val_loss"], criterion(logits, targets).item(), places=6)

    def test_batchmean_uses_output_batch_size_for_broadcast_targets(self):
        logits = torch.tensor([[1., 0., 2.], [3., 0., 1.], [2., 0., 4.]]).log_softmax(dim=1)
        targets = torch.tensor([[0.2, 0.5, 0.3]])
        loader = [(logits[:2], targets), (logits[2:], targets)]
        criterion = nn.KLDivLoss(reduction="batchmean")
        self.assertAlmostEqual(evaluate(nn.Identity(), loader, criterion, device="cpu"),
                               criterion(logits, targets).item(), places=6)

    def test_custom_prepare_batch_retains_structured_target_support(self):
        inputs = torch.tensor([[0.], [0.], [3.]])
        loader = [(inputs[:2], {"expected": torch.zeros(2, 1)}),
                  (inputs[2:], {"expected": torch.zeros(1, 1)})]
        criterion = lambda outputs, targets: (outputs - targets["expected"]).square().mean()
        prepare = lambda batch, device: batch
        self.assertEqual(evaluate(nn.Identity(), loader, criterion, device="cpu", prepare_batch=prepare), 3.0)
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1)
        trainer = Trainer(model, criterion, torch.optim.SGD(model.parameters(), lr=0),
                          device="cpu", prepare_batch=prepare)
        self.assertEqual(trainer._train_epoch(loader, show_progress=False), 3.0)

    def test_empty_validation_never_creates_a_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = nn.Linear(1, 1)
            trainer = Trainer(model, nn.MSELoss(), torch.optim.SGD(model.parameters(), lr=0.1), device="cpu")
            with patch.object(trainer, "_train_epoch", return_value=0.2), \
                    self.assertRaisesRegex(ValueError, "empty loader"):
                trainer.fit([], [], epochs=1, checkpoint=Checkpoint(tmpdir), progress=False)
            self.assertFalse((Path(tmpdir) / "best.pt").exists())


class TestExperimentRegressions(unittest.TestCase):
    def test_interrupt_preserves_caller_owned_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "existing.txt"
            path.write_text("keep this", encoding="utf-8")
            with patch("bokete.experiments.reporting.config_report"):
                result = run_experiments({"experiment_name": "interrupted"},
                                         run_fn=lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
                                         output_dir=tmpdir)
            self.assertEqual(len(result), 1)
            self.assertTrue(result[0]["interrupted"])
            self.assertTrue(path.exists())
            self.assertEqual(path.read_text(encoding="utf-8"), "keep this")

    def test_interrupt_preserves_last_completed_epoch(self):
        runner = create_trial_runner(lambda cfg: nn.Linear(1, 1), lambda cfg: ([], []))
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(Trainer, "_train_epoch", return_value=0.2), \
                    patch.object(Trainer, "evaluate", side_effect=[0.5, KeyboardInterrupt()]), \
                    self.assertRaises(KeyboardInterrupt):
                runner(1, {"experiment_name": "interrupted", "training": {"epochs": 2}}, output_dir=tmpdir)
            path = Path(tmpdir) / "trial_1" / "checkpoints" / "last.pt"
            self.assertTrue(path.exists())
            self.assertEqual(torch.load(path, weights_only=True)["epoch"], 1)

    def test_sweep_labels_cannot_merge_distinct_configurations(self):
        def runner(trial, config, output_dir, **kwargs):
            (Path(output_dir) / "marker.txt").write_text(config["data"]["path"], encoding="utf-8")
            return {"train_loss": [0.2], "val_loss": [0.3]}

        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_experiments({"experiment_name": "collision"}, run_fn=runner, output_dir=tmpdir,
                                      param_grid={"data.path": ["a/b", "a-b"]})
            paths = [Path(result["run_dir"]) for result in results]
            self.assertEqual(len(set(paths)), 2)
            self.assertEqual([(path / "marker.txt").read_text(encoding="utf-8") for path in paths], ["a/b", "a-b"])

    def test_concurrent_runs_claim_distinct_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir) / "concurrent"
            parent.mkdir()
            barrier = Barrier(2)
            original = Path.iterdir

            def synchronised_scan(path):
                entries = list(original(path))
                if path == parent:
                    barrier.wait(timeout=10)
                return iter(entries)

            with patch.object(Path, "iterdir", synchronised_scan), ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(create_run_directory, base_dir=tmpdir,
                                           experiment_name="concurrent", attach_file_logger=False) for _ in range(2)]
                paths = [future.result(timeout=15) for future in futures]
            self.assertEqual(len(set(paths)), 2)
            self.assertTrue(all(path.is_dir() for path in paths))


class TestConfigRegressions(unittest.TestCase):
    def test_explicit_path_wins_over_configs_basename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "configs").mkdir()
            (root / "elsewhere").mkdir()
            local = root / "configs" / "same.yaml"
            explicit = root / "elsewhere" / "same.yaml"
            local.write_text("experiment_name: wrong\n", encoding="utf-8")
            explicit.write_text("experiment_name: intended\n", encoding="utf-8")
            # Resolve both relative candidates against an isolated working directory.
            import os
            previous = Path.cwd()
            try:
                os.chdir(root)
                self.assertEqual(load_config(explicit)["experiment_name"], "intended")
                self.assertEqual(resolve_config_path("same"), Path("configs/same.yaml"))
                self.assertEqual(resolve_config_path("missing/same.yaml"), Path("missing/same.yaml"))
            finally:
                os.chdir(previous)

    def test_explicit_null_and_deferred_nested_annotations(self):
        with patch("bokete.utils.load_config", return_value={"training": {"epochs": 7, "patience": None}}):
            config = load_config_as("unused.yaml", _DeferredConfig)
        self.assertIsInstance(config.training, _TrainingConfig)
        self.assertEqual(config.training.epochs, 7)
        self.assertIsNone(config.training.patience)
        with patch("bokete.utils.load_config", return_value={}):
            self.assertEqual(load_config_as("unused.yaml", _DeferredConfig).training.patience, 3)


class TestScaffoldRegressions(unittest.TestCase):
    def test_generated_template_is_portable_and_excludes_runtime_files(self):
        manifest = {"main.py", "pyproject.toml", ".gitignore", "src/__init__.py", "src/models.py",
                    "src/schema.py", "configs/example.yaml"}
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "package" / "example"
            for name in manifest:
                destination = source / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / "example" / name, destination)
            for name in ("uv.lock", ".venv/marker", "data/marker", "results/marker", "src/__pycache__/marker.pyc"):
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("local artefact", encoding="utf-8")
            work = root / "workspace"
            work.mkdir()
            with patch("bokete.cli.__file__", str(root / "package" / "cli.py")), \
                    patch("bokete.cli.Path.cwd", return_value=work), \
                    patch("sys.stdout", new_callable=io.StringIO) as output:
                init_project("projects/demo")
            generated = work / "projects" / "demo"
            files = {path.relative_to(generated).as_posix() for path in generated.rglob("*") if path.is_file()}
            self.assertEqual(files, manifest)
            project = (generated / "pyproject.toml").read_text(encoding="utf-8")
            self.assertIn('name = "demo"', project)
            self.assertNotIn('path = ".."', project)
            self.assertIn(f'"bokete>={bokete.__version__}"', project)
            self.assertIn("uv run main.py --config example", output.getvalue())


class TestReportingRegressions(unittest.TestCase):
    def test_partial_histories_do_not_crash_configuration_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = config_report({}, [{"train_loss": [1.], "val_loss": [2.]}, {"train_loss": [3.]}], save_path=tmpdir)
            self.assertTrue((Path(tmpdir) / "config-report.md").exists())
            self.assertTrue((Path(tmpdir) / "multi_trial_loss.png").exists())
            self.assertIn("Trial 2", report)

    def test_readme_report_call_accepts_its_documented_arguments(self):
        calls = []
        for block in re.findall(r"```python\n(.*?)```", (ROOT / "README.md").read_text(encoding="utf-8"), re.DOTALL):
            for node in ast.walk(ast.parse(block)):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr in ("trial_report", "experiment_report")
                        and any(keyword.arg == "metrics" for keyword in node.keywords)):
                    calls.append(node)
        self.assertEqual(len(calls), 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            for keyword in calls[0].keywords:
                if keyword.arg == "save_path":
                    keyword.value = ast.Constant(str(Path(tmpdir) / "report.md"))
            expression = ast.fix_missing_locations(ast.Expression(calls[0]))
            report = eval(compile(expression, "README.md", "eval"), {
                "bokete": bokete, "config": {},
                "history": TrainingMetrics(train_loss=[0.2], val_loss=[0.3]),
            })
            self.assertIn("Final Validation Loss", report)
            self.assertTrue((Path(tmpdir) / "report.md").exists())
