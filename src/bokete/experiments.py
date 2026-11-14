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

import bokete.reporting as reporting
import bokete.metrics as metrics_mod
from bokete.trial_runner import create_trial_runner
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

    configured_trials = target_config.get("trials", 1) if isinstance(target_config, dict) else getattr(target_config, "trials", 1)
    n_trials = num_trials if num_trials is not None else configured_trials
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError("num_trials must be a positive integer.")
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

        resume_maps = _sweep_resume_maps(base_cfg, total, n_trials)

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


def _sweep_resume_maps(config, total_configurations, num_trials):
    """Validate and group sweep checkpoint paths by configuration and trial."""
    resume_maps = {}
    resume_config = asdict(config) if is_dataclass(config) else config
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
            if configuration > total_configurations or trial > num_trials:
                raise ValueError(f"training.{key} addresses a configuration or trial outside this sweep.")
            scoped.setdefault(configuration, {})[trial] = path
        resume_maps[key] = scoped

    return resume_maps
