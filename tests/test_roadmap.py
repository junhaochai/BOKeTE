"""Regression coverage for interrupted trials, experiment resume and reproducibility."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bokete import Trainer, TrainingInterrupted, create_trial_runner, run_experiments, set_seed, training_report
from bokete.reporting import config_report, experiment_report, trial_report
from bokete.utils import runtime_metadata


class TestRoadmap(unittest.TestCase):
    def setUp(self):
        algorithms = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        cudnn = torch.backends.cudnn.deterministic
        benchmark = torch.backends.cudnn.benchmark
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        self.addCleanup(torch.use_deterministic_algorithms, algorithms, warn_only=warn_only)
        self.addCleanup(setattr, torch.backends.cudnn, "deterministic", cudnn)
        self.addCleanup(setattr, torch.backends.cudnn, "benchmark", benchmark)
        set_seed(42, deterministic=False)

    def runner(self, **kwargs):
        def loaders(config):
            dataset = TensorDataset(torch.arange(16).reshape(8, 2).float() / 16,
                                    torch.arange(8).reshape(8, 1).float() / 8)

            loader = DataLoader(dataset, batch_size=3, shuffle=True,
                                generator=torch.Generator().manual_seed(11))

            return loader, DataLoader(dataset, batch_size=4)

        return create_trial_runner(lambda config: nn.Linear(2, 1), loaders,
                                   log_model_structure=False, **kwargs)

    def config(self, epochs=2):
        return {"experiment_name": "resume", "seed": 42,
                "training": {"epochs": epochs, "loss": "MSELoss"}}

    def test_interrupted_second_trial_retains_completed_and_partial_results(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(Trainer, "_train_epoch", return_value=0.2), \
                    patch.object(Trainer, "evaluate", side_effect=[0.8, 0.7, 0.6, KeyboardInterrupt()]):
                results = run_experiments(self.config(), run_fn=self.runner(), num_trials=3, output_dir=root)

            self.assertEqual([r["status"] for r in results], ["completed", "interrupted"])
            self.assertEqual(results[1]["train_loss"], [0.2])
            self.assertEqual(results[1]["val_loss"], [0.6])
            checkpoint = torch.load(Path(root) / "trial_2/checkpoints/last.pt", weights_only=True)

            self.assertEqual(checkpoint["epoch"], 1)
            self.assertFalse((Path(root) / "trial_3").exists())
            report = (Path(root) / "config-report.md").read_text(encoding="utf-8")

            self.assertIn("**Mean Best Validation Loss:** `0.7000", report)
            self.assertIn("**Status:** `interrupted`", report)
            self.assertIn("`interrupted`", (Path(root) / "trial_2/report.md").read_text(encoding="utf-8"))
            config = self.config()
            config["training"]["resume_from"] = {
                str(trial): str(Path(root) / f"trial_{trial}/checkpoints/last.pt") for trial in (1, 2)
            }

            with patch.object(Trainer, "_train_epoch", return_value=0.3) as train, \
                    patch.object(Trainer, "evaluate", return_value=0.9):
                resumed = run_experiments(config, run_fn=self.runner(), num_trials=3,
                                          output_dir=Path(root) / "resumed")

            self.assertEqual([r["status"] for r in resumed], ["completed"] * 3)
            self.assertEqual(resumed[0]["val_loss"], [0.8, 0.7])
            self.assertEqual(resumed[1]["val_loss"], [0.6, 0.9])
            self.assertEqual([call.kwargs["epoch_idx"] for call in train.call_args_list], [2, 1, 2])

    def test_sweep_retains_current_configuration_and_stops(self):
        calls = []

        def runner(trial, config, **kwargs):
            calls.append((config["training"]["lr"], trial))
            if len(calls) == 4:
                raise KeyboardInterrupt()
            return {"train_loss": [0.2], "val_loss": [0.3]}

        config = self.config()
        config["param_grid"] = {"training.lr": [0.1, 0.01, 0.001]}

        with tempfile.TemporaryDirectory() as root:
            results = run_experiments(config, run_fn=runner, num_trials=2, output_dir=root)
            report = (Path(root) / "experiment-report.md").read_text(encoding="utf-8")

        self.assertEqual(len(calls), 4)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(results[1]["metrics"]), 2)
        self.assertEqual(results[1]["status"], "interrupted")
        self.assertIn("**Interrupted Trials:** `1`", report)
        self.assertIn("**Status:** `interrupted`", report)

    def test_direct_fit_raises_with_complete_epoch_history(self):
        model = nn.Linear(2, 1)
        trainer = Trainer(model, nn.MSELoss(), torch.optim.SGD(model.parameters(), lr=0.1), device="cpu")

        with patch.object(trainer, "_train_epoch", return_value=0.2), \
                patch.object(trainer, "evaluate", side_effect=[0.3, KeyboardInterrupt()]), \
                self.assertRaises(TrainingInterrupted) as raised:
            trainer.fit([], [], epochs=3, progress=False)

        self.assertEqual(raised.exception.history.train_loss, [0.2])
        self.assertEqual(raised.exception.history.val_loss, [0.3])
        self.assertTrue(raised.exception.history.interrupted)

    def test_interrupting_config_report_retains_finished_trials(self):
        for sweep in (False, True):
            with self.subTest(sweep=sweep), tempfile.TemporaryDirectory() as root:
                config = self.config()
                if sweep:
                    config["param_grid"] = {"training.lr": [0.1, 0.01]}

                with patch("bokete.experiments.reporting.config_report", side_effect=KeyboardInterrupt()):
                    result = run_experiments(config, run_fn=lambda *args, **kwargs:
                                             {"train_loss": [0.2], "val_loss": [0.3]},
                                             num_trials=2, output_dir=root)

                histories = result[0]["metrics"] if sweep else result

                self.assertEqual(len(histories), 2)
                self.assertEqual(histories[0]["val_loss"], [0.3])
                if sweep:
                    self.assertEqual(result[0]["status"], "interrupted")

    def test_interrupt_before_first_epoch_has_report_without_broken_plot(self):
        runner = create_trial_runner(lambda config: nn.Linear(2, 1),
                                     lambda config: (_ for _ in ()).throw(KeyboardInterrupt()))

        with tempfile.TemporaryDirectory() as root:
            result = run_experiments(self.config(), run_fn=runner, output_dir=root)
            report = (Path(root) / "trial_1/report.md").read_text(encoding="utf-8")
            aggregate = (Path(root) / "config-report.md").read_text(encoding="utf-8")

        self.assertEqual(result[0]["status"], "interrupted")
        self.assertNotIn("![Loss Curve]", report)
        self.assertIn("No completed trial scores", aggregate)

    def test_scheduler_resume_matches_uninterrupted_runner(self):
        runner = self.runner(scheduler_factory=lambda optimizer, config:
                             torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5))

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            full = runner(1, self.config(4), output_dir=root / "full")
            runner(1, self.config(2), output_dir=root / "split")
            config = self.config(4)
            config["training"]["resume_from"] = {1: str(root / "split/trial_1/checkpoints/last.pt")}
            resumed = runner(1, config, output_dir=root / "resumed")
            full_state = torch.load(root / "full/trial_1/checkpoints/last.pt", weights_only=True)
            resumed_state = torch.load(root / "resumed/trial_1/checkpoints/last.pt", weights_only=True)

        self.assertEqual(full["train_loss"], resumed["train_loss"])
        self.assertEqual(full["val_loss"], resumed["val_loss"])
        self.assertEqual(full_state["training_state"]["callback_states"],
                         resumed_state["training_state"]["callback_states"])

        self.assertEqual(full_state["optimizer_state_dict"]["param_groups"],
                         resumed_state["optimizer_state_dict"]["param_groups"])

        for key, value in full_state["model_state_dict"].items():
            self.assertTrue(torch.equal(value, resumed_state["model_state_dict"][key]))

    def test_trials_resume_their_own_weights_without_extra_epochs(self):
        runner = self.runner(test_fn=lambda model, config: {"weight": float(model.weight.detach()[0, 0])})

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            first = [runner(trial, self.config(1), output_dir=root / "original") for trial in (1, 2)]
            config = self.config(1)
            config["training"]["resume_from"] = {
                str(trial): str(root / f"original/trial_{trial}/checkpoints/last.pt") for trial in (1, 2)
            }

            with patch.object(Trainer, "_train_epoch", side_effect=AssertionError("target already reached")):
                resumed = [runner(trial, config, output_dir=root / "resumed") for trial in (1, 2)]

        self.assertNotEqual(first[0]["extra_metrics"]["Final weight"], first[1]["extra_metrics"]["Final weight"])
        self.assertEqual([r["extra_metrics"] for r in first], [r["extra_metrics"] for r in resumed])

    def test_original_best_survives_new_directory_and_second_resume(self):
        runner = self.runner(test_fn=lambda model, config: {"weight": float(model.weight.detach()[0, 0])})

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with patch.object(Trainer, "evaluate", side_effect=[0.1, 0.5]):
                original = runner(1, self.config(2), output_dir=root / "original")

            config = self.config(3)
            config["training"]["resume_from"] = {1: str(root / "original/trial_1/checkpoints/last.pt")}

            with patch.object(Trainer, "evaluate", return_value=0.6):
                resumed = runner(1, config, output_dir=root / "resumed")

            self.assertEqual(original["extra_metrics"]["Best weight"], resumed["extra_metrics"]["Best weight"])
            self.assertNotEqual(resumed["extra_metrics"]["Best weight"], resumed["extra_metrics"]["Final weight"])
            retained = torch.load(root / "resumed/trial_1/checkpoints/best.pt", weights_only=True)

            self.assertEqual(retained["epoch"], 1)
            config["training"]["epochs"] = 4
            config["training"]["resume_from"] = {1: str(root / "resumed/trial_1/checkpoints/last.pt")}

            with patch.object(Trainer, "evaluate", return_value=0.7):
                again = runner(1, config, output_dir=root / "again")

            self.assertEqual(original["extra_metrics"]["Best weight"], again["extra_metrics"]["Best weight"])

    def test_disabled_resume_checkpoints_only_reuse_unchanged_best(self):
        for improved in (False, True):
            for evaluate_test in (False, True):
                with self.subTest(improved=improved, evaluate_test=evaluate_test), \
                        tempfile.TemporaryDirectory() as root:
                    root = Path(root)
                    test_fn = (lambda model, config: {"weight": float(model.weight.detach()[0, 0])}) \
                        if evaluate_test else None

                    runner = self.runner(test_fn=test_fn)

                    with patch.object(Trainer, "evaluate", side_effect=[0.5, 0.6]):
                        original = runner(1, self.config(2), output_dir=root / "original")

                    config = self.config(3)
                    config["training"]["save_checkpoints"] = False
                    config["training"]["resume_from"] = {
                        1: str(root / "original/trial_1/checkpoints/last.pt"),
                    }

                    with patch.object(Trainer, "evaluate", return_value=0.4 if improved else 0.7):
                        resumed = runner(1, config, output_dir=root / "resumed")

                    self.assertEqual(resumed["best_epoch"], 3 if improved else 1)
                    self.assertEqual(resumed["best_checkpoint_available"], not improved)
                    self.assertFalse((root / "resumed/trial_1/checkpoints/best.pt").exists())
                    if evaluate_test:
                        self.assertIn("Final weight", resumed["extra_metrics"])
                        if improved:
                            self.assertNotIn("Best weight", resumed["extra_metrics"])
                        else:
                            self.assertEqual(resumed["extra_metrics"]["Best weight"],
                                             original["extra_metrics"]["Best weight"])
                    else:
                        self.assertNotIn("extra_metrics", resumed)

                    report = (root / "resumed/trial_1/report.md").read_text(encoding="utf-8")

                    self.assertEqual("**Best Checkpoint:** unavailable" in report, improved)

    def test_missing_best_is_unavailable_and_explicit_mismatch_is_rejected(self):
        runner = self.runner(test_fn=lambda model, config: {"weight": float(model.weight.detach()[0, 0])})

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with patch.object(Trainer, "evaluate", side_effect=[0.1, 0.5]):
                runner(1, self.config(2), output_dir=root / "original")

            directory = root / "original/trial_1/checkpoints"
            (directory / "best.pt").rename(directory / "archived-best.pt")
            config = self.config(2)
            config["training"]["resume_from"] = {1: str(directory / "last.pt")}
            result = runner(1, config, output_dir=root / "missing")

            self.assertFalse(result["best_checkpoint_available"])
            self.assertNotIn("Best weight", result["extra_metrics"])
            report = (root / "missing/trial_1/report.md").read_text(encoding="utf-8")

            self.assertIn("**Best Checkpoint:** unavailable", report)
            config["training"]["resume_best_from"] = {1: str(directory / "last.pt")}

            with self.assertRaisesRegex(ValueError, "does not match"):
                runner(1, config, output_dir=root / "mismatch")

            config["training"]["resume_best_from"] = {1: str(directory / "archived-best.pt")}
            restored = runner(1, config, output_dir=root / "explicit")

            self.assertTrue(restored["best_checkpoint_available"])
            self.assertIn("Best weight", restored["extra_metrics"])

    def test_resume_paths_must_be_per_trial(self):
        config = self.config()
        config["training"]["resume_from"] = "one-checkpoint-for-everyone.pt"

        with self.assertRaisesRegex(ValueError, "must map trial numbers"):
            self.runner()(1, config)

    def test_deterministic_mode_can_be_enabled_then_disabled(self):
        set_seed(42, deterministic=True)

        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)
        set_seed(42, deterministic=False)

        self.assertFalse(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.backends.cudnn.deterministic)

    def test_benchmark_policy_respects_determinism(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                set_seed(42, deterministic=enabled)
                model = nn.Linear(2, 1)
                observed = []

                def criterion(outputs, targets):
                    observed.append(torch.backends.cudnn.benchmark)
                    return nn.functional.mse_loss(outputs, targets)

                trainer = Trainer(model, criterion, torch.optim.SGD(model.parameters(), lr=0.1), device="cpu",
                                  prepare_batch=lambda batch, device: batch)

                # Exercise CUDA policy decisions with CPU tensors; actual CUDA resume has its own hardware test.
                trainer.device = torch.device("cuda")
                trainer._train_epoch([(torch.ones(2, 2), torch.ones(2, 1))], show_progress=False)

                self.assertEqual(observed, [not enabled])

    def test_runner_applies_deterministic_setting_before_factories(self):
        observed = []

        def loaders(config):
            observed.append(torch.are_deterministic_algorithms_enabled())
            raise KeyboardInterrupt()

        runner = create_trial_runner(lambda config: nn.Linear(2, 1), loaders)

        with tempfile.TemporaryDirectory() as root:
            for enabled in (True, False):
                config = self.config()
                config["deterministic"] = enabled
                run_experiments(config, run_fn=runner, output_dir=Path(root) / str(enabled))

        self.assertEqual(observed, [True, False])

    def test_sweep_resume_paths_are_scoped_to_configuration_and_trial(self):
        config = self.config()
        config["param_grid"] = {"training.lr": [0.1, 0.01]}
        config["training"]["resume_from"] = {"1:2": "first-trial-two.pt", "2:1": "second-trial-one.pt"}
        config["training"]["resume_best_from"] = {"2:1": "second-best.pt"}
        observed = []

        def runner(trial, cfg, **kwargs):
            observed.append((trial, cfg["training"]["resume_from"], cfg["training"]["resume_best_from"]))
            return {"train_loss": [0.2], "val_loss": [0.3]}

        with tempfile.TemporaryDirectory() as root:
            run_experiments(config, run_fn=runner, num_trials=2, output_dir=root)

        self.assertEqual(observed, [(1, {2: "first-trial-two.pt"}, {}), (2, {2: "first-trial-two.pt"}, {}),
                                    (1, {1: "second-trial-one.pt"}, {1: "second-best.pt"}),
                                    (2, {1: "second-trial-one.pt"}, {1: "second-best.pt"})])

        config["training"]["resume_from"] = {1: "ambiguous.pt"}

        with tempfile.TemporaryDirectory() as root, self.assertRaisesRegex(ValueError, "configuration:trial"):
            run_experiments(config, run_fn=runner, output_dir=root)

    def test_resume_rejects_changed_deterministic_policy(self):
        with tempfile.TemporaryDirectory() as root:
            config = self.config(1)
            config["deterministic"] = True
            self.runner()(1, config, output_dir=root)
            config["deterministic"] = False
            config["training"]["resume_from"] = {1: str(Path(root) / "trial_1/checkpoints/last.pt")}

            with self.assertRaisesRegex(ValueError, "same deterministic settings"):
                self.runner()(1, config, output_dir=Path(root) / "resumed")

    def test_model_factory_runs_only_after_deterministic_setup(self):
        observed = []

        def model_factory(config):
            observed.append(torch.are_deterministic_algorithms_enabled())
            return nn.Linear(2, 1)

        runner = create_trial_runner(model_factory, lambda config: ([], []))
        config = self.config(1)
        config["deterministic"] = True

        with tempfile.TemporaryDirectory() as root:
            with patch.object(Trainer, "_train_epoch", return_value=0.2), \
                    patch.object(Trainer, "evaluate", return_value=0.3):
                run_experiments(config, run_fn=runner, num_trials=2, output_dir=root)

            report = (Path(root) / "config-report.md").read_text(encoding="utf-8")

        self.assertEqual(observed, [True, True])
        self.assertIn("**Model Class:** `Linear`", report)

    def test_reports_use_recorded_runtime_and_exclude_interrupted_scores(self):
        runtime = {**runtime_metadata(), "pytorch": "recorded-version"}
        completed = {"train_loss": [0.2], "val_loss": [0.5], "runtime": runtime, "trial": 1}
        interrupted = {"train_loss": [0.1], "val_loss": [0.01], "interrupted": True,
                       "runtime": runtime, "trial": 2, "extra_metrics": {"accuracy": 1.0}}

        configuration = config_report({}, [completed, interrupted], auto_plot=False)

        self.assertIn("**Mean Best Validation Loss:** `0.5000", configuration)
        self.assertNotIn("**Mean Accuracy:**", configuration)
        sweep = experiment_report({}, [{"parameters": {"lr": 0.1}, "metrics": [completed]},
                                       {"parameters": {"lr": 0.2}, "metrics": [interrupted]}])

        self.assertIn("**Configuration:** `lr=0.1`", sweep)
        self.assertIn("| N/A | `lr=0.2`", sweep)

        for report in (configuration, sweep, trial_report({}, metrics=completed, auto_plot=False),
                       trial_report({}, metrics_summary=training_report(completed), auto_plot=False)):
            self.assertIn("`recorded-version`", report)
            self.assertIn("## Runtime Environment", report)

        self.assertNotIn("recorded-version", trial_report({}, metrics={"train_loss": [], "val_loss": [],
                                                                       "interrupted": True}, auto_plot=False))


if __name__ == "__main__":
    unittest.main()
