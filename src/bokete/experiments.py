"""
Orchestration of hyperparameter sweeps and multi-trial experimental runs.

Key Functions:
  - run_experiments   : Executes single-config multi-trial runs or hyperparameter grid sweeps.
  - create_trial_runner: Helper factory that builds a standardised single-trial runner callback.
"""

import copy
from dataclasses import asdict, is_dataclass
import itertools
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

import bokete.reporting as reporting
import bokete.metrics as metrics_mod
import bokete.training as training
import bokete.utils as utils


logger = logging.getLogger(__name__)


def _run_trials(
    config: Dict[str, Any],
    num_trials: int,
    run_fn: Callable[..., Dict[str, Any]],
    output_dir: Optional[str] = None,
    model: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Executes all trials for a configuration, aggregating metrics, loss plots, and reports."""
    exp_label = config.get("experiment_name") if isinstance(config, dict) else getattr(config, "experiment_name", None)
    if not exp_label or not str(exp_label).strip():
        raise ValueError("Configuration must explicitly define 'experiment_name'.")
    monitor_config = asdict(config) if is_dataclass(config) else config
    configured_monitor = utils.get_nested_key(monitor_config, "training.monitor")
    if configured_monitor is not None:
        metrics_mod.monitor_mode(configured_monitor)

    if output_dir is None:
        run_dir_path = utils.create_run_directory(config, attach_file_logger=True)
        output_dir = str(run_dir_path)
    else:
        utils.setup_logging(log_file=Path(output_dir) / "experiment.log")

    all_trial_metrics = []

    start_dt = datetime.now()
    start_time = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.time()

    # Use the actual trial model's metadata; do not initialise CUDA before trial seeding.
    active_model = model

    try:
        for i in range(1, num_trials + 1):
            try:
                metrics = run_fn(i, config, output_dir=output_dir, total_trials=num_trials)
            except KeyboardInterrupt as exc:
                history = getattr(exc, "history", None)
                metrics = (history.as_dict() if history is not None else
                           metrics_mod.TrainingMetrics(
                               monitor=configured_monitor or "val_loss", interrupted=True,
                               runtime=utils.runtime_metadata(),
                           ).as_dict())
                metrics["interrupted"] = True
            if metrics:
                if active_model is None:
                    active_model = getattr(run_fn, "model_metadata", None)
                metrics = dict(metrics)
                metrics["trial"] = i
                metrics["status"] = "interrupted" if metrics.get("interrupted") else "completed"
                metrics["run_dir"] = str(Path(output_dir) / f"trial_{i}")
            if metrics:
                all_trial_metrics.append(metrics)
                if metrics.get("interrupted"):
                    logger.warning("[BOKeTE] Trial %s interrupted; stopping remaining trials.", i)
                    break
                selection = metrics_mod.selection_summary(metrics, configured_monitor)
                if selection['best_score'] is not None:
                    best_val = selection['best_score']
                    metric_label = utils.format_metric_name(selection['monitor'])
                    logger.info("")
                    logger.info(f"[BOKeTE] Trial {i} complete — Best {metric_label}: {utils.Colours.BRIGHT_CYAN}{best_val:.4f}{utils.Colours.RESET}")
                else:
                    logger.info("")
                    logger.info(f"[BOKeTE] Trial {i} complete")
        t1 = time.time()
        end_dt = datetime.now()
        end_time = end_dt.strftime("%Y-%m-%d %H:%M:%S")
        duration_seconds = round(t1 - t0, 2)

        # Guard: Ensure output directory exists and metrics were produced before generating report
        # (prevents errors on empty runs)
        if output_dir and all_trial_metrics:
            logger.info("")
            reporting.config_report(
                config=config,
                all_trial_metrics=all_trial_metrics,
                save_path=output_dir,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration_seconds,
                model=active_model,
            )

        # Reuse the monitored scores already recorded by each trial.
        selections = [metrics_mod.selection_summary(m, configured_monitor) for m in all_trial_metrics
                      if not m.get("interrupted")]
        best_val_losses = [s['best_score'] for s in selections if s['best_score'] is not None]
        if best_val_losses:
            mean_val = float(np.mean(best_val_losses))
            std_val = float(np.std(best_val_losses))
            min_val = float(np.min(best_val_losses))
            max_val = float(np.max(best_val_losses))
            summary_label = f" for {utils.format_tag(exp_label)}" if exp_label else ""
            logger.info("")
            logger.info(
                f"[BOKeTE] === Experiment {'Interrupted' if any(m.get('interrupted') for m in all_trial_metrics) else 'Complete'}{summary_label} ({len(best_val_losses)} Completed Trials) | "
                f"Time Taken: {utils.Colours.BRIGHT_CYAN}{utils.format_duration(duration_seconds)}{utils.Colours.RESET} "
                f"(Start: {start_time}, End: {end_time}) ==="
            )
            logger.info(
                f"[BOKeTE] Mean Best {utils.format_metric_name(selections[0]['monitor'])}: {utils.Colours.BRIGHT_CYAN}{mean_val:.4f} ± {std_val:.4f}{utils.Colours.RESET} "
                f"(Min: {utils.Colours.CYAN}{min_val:.4f}{utils.Colours.RESET}, Max: {utils.Colours.CYAN}{max_val:.4f}{utils.Colours.RESET})"
            )
            logger.info("")
    except KeyboardInterrupt as exc:
        # Report generation can also be interrupted after trials have finished.
        exc.trial_metrics = all_trial_metrics
        raise
    finally:
        utils.close_file_loggers()

    return all_trial_metrics


def run_experiments(
    config: Optional[Dict[str, Any]] = None,
    run_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    num_trials: Optional[int] = None,
    output_dir: Optional[str] = None,
    model: Optional[Any] = None,
    base_config: Optional[Dict[str, Any]] = None,
    param_grid: Optional[Dict[str, List[Any]]] = None,
) -> List[Dict[str, Any]]:
    """Primary top-level entry point to execute an experiment run.

    Automatically detects whether to perform a multi-config hyperparameter sweep (if param_grid
    is present in config or arguments) or a single-config multi-trial run.
    """
    # Support legacy positional signature: run_experiments(base_config, param_grid, run_fn)
    if isinstance(run_fn, dict) and param_grid is None:
        param_grid = run_fn
        run_fn = num_trials  # type: ignore
        num_trials = None

    target_config = config if config is not None else base_config
    if target_config is None:
        raise ValueError("A configuration dictionary must be provided to run_experiments().")

    # Initialise console logger immediately so pre-trial banners are never dropped
    utils.setup_logging()

    grid = param_grid
    if grid is None and isinstance(target_config, dict):
        grid = target_config.get("param_grid")
    elif grid is None and is_dataclass(target_config):
        grid = getattr(target_config, "param_grid", None)

    n_trials = num_trials or (target_config.get("trials", 1) if isinstance(target_config, dict) else getattr(target_config, "trials", 1))
    exp_name = target_config.get("experiment_name") if isinstance(target_config, dict) else getattr(target_config, "experiment_name", None)

    if grid:
        keys = list(grid.keys())
        value_combinations = list(itertools.product(*grid.values()))
        results = []
        total = len(value_combinations)

        base_cfg = copy.deepcopy(asdict(target_config) if is_dataclass(target_config) else target_config)
        if isinstance(base_cfg, dict):
            base_cfg.pop("param_grid", None)

        sweep_start_dt = datetime.now()
        sweep_start_time = sweep_start_dt.strftime("%Y-%m-%d %H:%M:%S")
        sweep_t0 = time.time()

        sweep_batch_dir = Path(output_dir) if output_dir else utils.create_run_directory(target_config, attach_file_logger=False)

        # Sweeps address checkpoints by configuration and trial, avoiding accidental reuse.
        resume_maps = {}
        resume_config = asdict(base_cfg) if is_dataclass(base_cfg) else base_cfg
        for key in ("resume_from", "resume_best_from"):
            paths = utils.get_nested_key(resume_config, f"training.{key}", default={})
            if not isinstance(paths, dict):
                raise ValueError(f"training.{key} must map configuration:trial keys to checkpoint paths in sweeps.")
            scoped = {}
            for label, path in paths.items():
                parts = str(label).split(":")
                if len(parts) != 2 or not all(part.isdecimal() and int(part) > 0 for part in parts):
                    raise ValueError(f"training.{key} requires 'configuration:trial' keys in sweeps.")
                configuration, trial = map(int, parts)
                if configuration > total or trial > n_trials:
                    raise ValueError(f"training.{key} addresses a configuration or trial outside this sweep.")
                scoped.setdefault(configuration, {})[trial] = path
            resume_maps[key] = scoped

        try:
            for idx, combo in enumerate(value_combinations, 1):
                run_config = copy.deepcopy(base_cfg)
                for key, scoped in resume_maps.items():
                    utils.set_nested_key(run_config, f"training.{key}", scoped.get(idx, {}))
                params_used = {}
                folder_parts = []
                for key, value in zip(keys, combo):
                    utils.set_nested_key(run_config, key, value)
                    params_used[key] = value
                    short_k = key.split(".")[-1]
                    clean_v = str(value).replace(" ", "").replace(":", "-").replace("/", "-").replace("\\", "")
                    folder_parts.append(f"{short_k}_{clean_v}")

                params_str = ", ".join([f"{k}={v}" for k, v in params_used.items()])
                utils.log_experiment_start(idx, total, exp_name, params_str=params_str)

                param_folder_name = "_".join(folder_parts) if folder_parts else f"config_{idx}"
                combo_dir = sweep_batch_dir / f"{idx:03d}_{param_folder_name}"
                combo_dir.mkdir(parents=True, exist_ok=True)

                trial_metrics = _run_trials(
                    config=run_config,
                    num_trials=n_trials,
                    run_fn=run_fn,
                    output_dir=str(combo_dir),
                    model=model,
                )
                results.append({
                    "parameters": params_used,
                    "metrics": trial_metrics,
                    "run_dir": str(combo_dir),
                    "status": "interrupted" if any(m.get("interrupted") for m in trial_metrics) else "completed",
                })
                if results[-1]["status"] == "interrupted":
                    break
        except KeyboardInterrupt as exc:
            utils.close_file_loggers()
            logger.warning("")
            logger.warning(f"[BOKeTE] Experiment sweep {utils.format_tag(exp_name)} aborted by user (Ctrl+C).")
            if hasattr(exc, "trial_metrics"):
                results.append({
                    "parameters": params_used,
                    "metrics": exc.trial_metrics,
                    "run_dir": str(combo_dir),
                    "status": "interrupted",
                })

        sweep_t1 = time.time()
        sweep_end_dt = datetime.now()
        sweep_end_time = sweep_end_dt.strftime("%Y-%m-%d %H:%M:%S")
        sweep_duration = round(sweep_t1 - sweep_t0, 2)

        total_trials_run = sum(len(r.get("metrics", [])) for r in results)
        logger.info("")
        logger.info(
            f"[BOKeTE] === Experiment Sweep {'Interrupted' if any(r.get('status') == 'interrupted' for r in results) else 'Complete'} for {utils.format_tag(exp_name)} "
            f"({len(results)} Configurations, {total_trials_run} Trials) | "
            f"Time Taken: {utils.Colours.BRIGHT_CYAN}{utils.format_duration(sweep_duration)}{utils.Colours.RESET} "
            f"(Start: {sweep_start_time}, End: {sweep_end_time}) ==="
        )

        reporting.experiment_report(
            config=target_config,
            sweep_results=results,
            save_path=sweep_batch_dir,
            start_time=sweep_start_time,
            end_time=sweep_end_time,
            duration_seconds=sweep_duration,
        )
        return results

    try:
        return _run_trials(
            config=target_config,
            num_trials=n_trials,
            run_fn=run_fn,
            output_dir=output_dir,
            model=model,
        )
    except KeyboardInterrupt as exc:
        utils.close_file_loggers()
        logger.warning("")
        logger.warning(f"[BOKeTE] Experiment {utils.format_tag(exp_name)} aborted by user (Ctrl+C).")
        return getattr(exc, "trial_metrics", [])


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
                lr = utils.get_nested_key(config, "training.lr") or 1e-3
                opt_cls = getattr(torch.optim, opt_name)
                optimizer = opt_cls(model.parameters(), lr=lr)

            scheduler = scheduler_factory(optimizer, config) if scheduler_factory else None

            # 5. Determine device & run Trainer (log device and structure on trial 1)
            is_first_trial = (trial_num == 1)
            device = utils.determine_device(verbose=is_first_trial)
            should_log_struct = log_model_structure and is_first_trial

            amp = bool(utils.get_nested_key(config, "training.amp", default=False))
            trainer = training.Trainer(
                model,
                criterion,
                optimizer,
                device=device,
                amp=amp,
                log_model_structure=should_log_struct,
            )

            epochs = utils.get_nested_key(config, "training.epochs") or 5
            patience = utils.get_nested_key(config, "training.patience")
            early_stopping_obj = training.EarlyStopping(patience=patience) if patience is not None and patience > 0 else None
            save_checkpoints = utils.get_nested_key(config, "training.save_checkpoints", default=True)

            log_interval = utils.get_nested_key(config, "training.log_interval") or 1

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

            # 6. Evaluate post-training test metrics if test_fn provided (for both Final and Best checkpoints)
            extra_metrics = {}
            if test_fn:
                def _eval_model(m):
                    return test_fn(m, test_loader, config) if test_loader is not None else test_fn(m, config)

                # 6a. Evaluate Final epoch weights (current in-memory state)
                raw_final_metrics = _eval_model(model) or {}
                for k, v in raw_final_metrics.items():
                    extra_metrics[f"Final {k}"] = v

                # 6b. Evaluate Best epoch weights (loaded from best.pt if available)
                best_ckpt_path = checkpoint.directory / "best.pt" if checkpoint.best_saved else original_best
                if best_ckpt_path is not None:
                    ckpt_data = torch.load(best_ckpt_path, map_location="cpu", weights_only=True)
                    raw_model = getattr(model, "module", model)
                    raw_model.load_state_dict(ckpt_data["model_state_dict"])
                    del ckpt_data
                    raw_best_metrics = _eval_model(model) or {}
                    for k, v in raw_best_metrics.items():
                        extra_metrics[f"Best {k}"] = v

            extra_metrics = extra_metrics if extra_metrics else None
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
