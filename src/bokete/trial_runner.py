"""Construction and execution of individual training trials."""

from dataclasses import asdict, is_dataclass
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch

import bokete.metrics as metrics_mod
import bokete.reporting as reporting
import bokete.training as training
import bokete.utils as utils


logger = logging.getLogger(__name__)


def create_trial_runner(
    model_factory: Callable[[Dict[str, Any]], Any],
    loader_factory: Callable[[Dict[str, Any]], Any],
    criterion_factory: Optional[Callable[[Dict[str, Any]], Any]] = None,
    optimizer_factory: Optional[Callable[[Any, Dict[str, Any]], Any]] = None,
    test_fn: Optional[Callable[[Any, Dict[str, Any]], Dict[str, Any]]] = None,
    log_model_structure: bool = True,
    scheduler_factory: Optional[Callable[[Any, Dict[str, Any]], Any]] = None,
) -> Callable[..., Dict[str, Any]]:
    """Helper factory that constructs a standardised single-trial runner callback for bokete.run_experiments.

    Args:
        model_factory: Callable `(config) -> torch.nn.Module`.
        loader_factory: Callable `(config) -> (train_loader, val_loader)`.
        criterion_factory: Optional callable `(config) -> criterion`.
        optimizer_factory: Optional callable `(model, config) -> optimizer`.
        test_fn: Optional callable `(model, config) -> dict` returning extra test metrics (e.g. Test Accuracy).
        log_model_structure: Whether to log the full PyTorch model layer structure on trial 1 (default: True).
        scheduler_factory: Optional callable `(optimizer, config) -> scheduler`.

    Returns:
        A callback `(trial_num, config, output_dir=None) -> metrics_dict` compatible with bokete.run_experiments.
    """
    def run_fn(trial_num: int, config: Dict[str, Any], output_dir: Optional[str] = None, total_trials: int = 1) -> Dict[str, Any]:
        run_fn.model_metadata = None

        config = asdict(config) if is_dataclass(config) else config
        exp_name = config.get("experiment_name") if isinstance(config, dict) else getattr(config, "experiment_name", None)
        if not exp_name or not str(exp_name).strip():
            raise ValueError("Configuration must explicitly define 'experiment_name'.")

        # 1. Set seed per trial for stochastic variance
        base_seed = config.get("seed", 42)
        trial_seed = base_seed + (trial_num - 1)

        monitor = utils.get_nested_key(config, "training.monitor", default="val_loss")
        metrics_mod.monitor_mode(monitor)

        utils.set_seed(trial_seed, deterministic=config.get("deterministic", False))

        resume_from = _trial_resume_path(config, "resume_from", trial_num)
        resume_best_from = _trial_resume_path(config, "resume_best_from", trial_num)
        if resume_best_from is not None and resume_from is None:
            raise ValueError("resume_best_from requires a resume_from path for the same trial.")

        if not output_dir:
            output_dir = str(utils.create_run_directory(config, attach_file_logger=False))

        trial_dir = Path(output_dir) / f"trial_{trial_num}"

        # 2. Log trial start header immediately so the user gets active feedback before data/model loading
        utils.log_trial_start(trial_num, total_trials, exp_name, seed=trial_seed)

        history = None
        try:
            # 3. Build dataloaders & model
            train_loader, val_loader, *test = loader_factory(config)
            test_loader = test[0] if test else None

            model = model_factory(config)
            run_fn.model_metadata = utils._model_info(model)

            # 4. Build criterion & optimizer
            if criterion_factory:
                criterion = criterion_factory(config)
            else:
                loss_name = utils.get_nested_key(config, "training.loss") or "CrossEntropyLoss"
                criterion = getattr(torch.nn, loss_name)()

            if optimizer_factory:
                optimizer = optimizer_factory(model, config)
            else:
                opt_name = utils.get_nested_key(config, "training.optimizer") or "Adam"
                lr = utils.get_nested_key(config, "training.lr", default=1e-3)
                opt_cls = getattr(torch.optim, opt_name)
                optimizer = opt_cls(model.parameters(), lr=lr)

            scheduler = scheduler_factory(optimizer, config) if scheduler_factory else None

            # 5. Determine device & run Trainer (log device and structure on trial 1)
            is_first_trial = (trial_num == 1)
            requested_device = config.get("device")
            device = (torch.device(requested_device) if requested_device is not None
                      else utils.determine_device(verbose=is_first_trial))

            should_log_struct = log_model_structure and is_first_trial

            amp = bool(utils.get_nested_key(config, "training.amp", default=False))
            trainer = training.Trainer(
                model,
                criterion,
                optimizer,
                device=device,
                amp=amp,
                max_grad_norm=utils.get_nested_key(config, "training.max_grad_norm"),
                log_model_structure=should_log_struct,
            )

            epochs = utils.get_nested_key(config, "training.epochs", default=5)
            patience = utils.get_nested_key(config, "training.patience")
            early_stopping_obj = training.EarlyStopping(patience=patience) if patience is not None and patience > 0 else None
            save_checkpoints = utils.get_nested_key(config, "training.save_checkpoints", default=True)

            log_interval = utils.get_nested_key(config, "training.log_interval", default=1)

            checkpoint = training.Checkpoint(trial_dir / "checkpoints", enabled=save_checkpoints)
            original_best = _retain_resume_best(resume_from, resume_best_from, checkpoint)
            history = trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                early_stopping=early_stopping_obj,
                checkpoint=checkpoint,
                log_interval=log_interval,
                monitor=monitor,
                scheduler=scheduler,
                resume_from=resume_from,
            )

            # Disabled writes leave the checkpoint score at the resumed selection.
            if (original_best is not None and not checkpoint.best_saved
                    and checkpoint.best_score != history.best_score):
                original_best = None

            # 6. Evaluate final weights, then best weights when available.
            best_ckpt_path = checkpoint.directory / "best.pt" if checkpoint.best_saved else original_best
            extra_metrics = _evaluate_test_metrics(
                model, test_fn, test_loader, config, best_ckpt_path,
            )

            result = history.as_dict()
            result["best_checkpoint_available"] = checkpoint.best_saved or original_best is not None
            if extra_metrics:
                result["extra_metrics"] = extra_metrics

            # 7. Generate per-trial Markdown report
            reporting.trial_report(
                config=config,
                metrics=result,
                model=model,
                extra_metrics=extra_metrics,
                save_path=trial_dir,
                title=f"{exp_name} - Trial {trial_num}",
            )

            return result

        except KeyboardInterrupt as exc:
            logger.warning("[BOKeTE] Trial interrupted; retaining files in %s", trial_dir)

            partial_history = getattr(exc, "history", history)
            if partial_history is None:
                partial_history = metrics_mod.TrainingMetrics(
                    monitor=monitor, interrupted=True, runtime=utils.runtime_metadata(),
                )
            partial_history.interrupted = True

            reporting.trial_report(
                config=config, metrics=partial_history, save_path=trial_dir,
                title=f"{exp_name} - Trial {trial_num}",
            )

            raise training.TrainingInterrupted(partial_history) from None

    run_fn.model_factory = model_factory  # type: ignore[attr-defined]
    return run_fn


def _trial_resume_path(config, key, trial_num):
    paths = utils.get_nested_key(config, f"training.{key}", default={})
    if not isinstance(paths, dict):
        raise ValueError(f"training.{key} must map trial numbers to checkpoint paths.")
    if any(not str(label).isdecimal() or int(label) < 1 for label in paths):
        raise ValueError(f"training.{key} requires positive trial numbers outside sweeps.")

    path = paths.get(trial_num, paths.get(str(trial_num)))
    return Path(path) if path is not None else None


def _retain_resume_best(resume_from, explicit_best, checkpoint):
    """Retain only best weights matching the selected epoch in the resume history."""
    if resume_from is None:
        return None

    saved = torch.load(resume_from, map_location="cpu", weights_only=True)
    state = saved.get("training_state")
    if state is None:
        raise ValueError("Checkpoint does not contain supported full training state.")

    selection = metrics_mod.selection_summary({**state["history"], "monitor": saved["monitor"]})
    source = explicit_best or (resume_from if saved["epoch"] == selection["best_epoch"]
                               else resume_from.with_name("best.pt"))

    del saved

    if not source.is_file():
        if explicit_best is not None:
            raise FileNotFoundError(source)
        logger.warning("[BOKeTE] Original best checkpoint unavailable: %s", source)
        return None

    best = torch.load(source, map_location="cpu", weights_only=True)
    if (best.get("epoch") != selection["best_epoch"]
            or best.get("score") != selection["best_score"]
            or best.get("monitor") != selection["monitor"]):
        if explicit_best is not None:
            raise ValueError("Original best checkpoint does not match the resumed selection.")
        logger.warning("[BOKeTE] Original best checkpoint does not match the resumed selection: %s", source)
        return None

    destination = checkpoint.directory / "best.pt"
    if checkpoint.enabled and source.resolve() != destination.resolve():
        checkpoint.directory.mkdir(parents=True, exist_ok=True)
        training._atomic_save(best, destination)
        return destination
    return source


def _evaluate_test_metrics(model, test_fn, test_loader, config, best_checkpoint):
    """Evaluate final and best weights, leaving best weights loaded when available."""
    if not test_fn:
        return None

    extra_metrics = {}

    def _eval_model(m):
        return test_fn(m, test_loader, config) if test_loader is not None else test_fn(m, config)

    # Evaluate final epoch weights (current in-memory state).
    raw_final_metrics = _eval_model(model) or {}
    for k, v in raw_final_metrics.items():
        extra_metrics[f"Final {k}"] = v

    # Evaluate best epoch weights (loaded from best.pt if available).
    if best_checkpoint is not None:
        ckpt_data = torch.load(best_checkpoint, map_location="cpu", weights_only=True)
        raw_model = getattr(model, "module", model)
        raw_model.load_state_dict(ckpt_data["model_state_dict"])
        del ckpt_data

        raw_best_metrics = _eval_model(model) or {}
        for k, v in raw_best_metrics.items():
            extra_metrics[f"Best {k}"] = v

    return extra_metrics or None
