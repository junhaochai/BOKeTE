"""
Orchestration of hyperparameter sweeps and multi-trial experimental runs.

Key Functions:
  - run_experiments   : Generates parameter grid combinations and executes run_fn for each.
  - run_trials        : Executes a single experiment configuration across multiple random seed trials.
  - create_trial_runner: Helper factory that builds a standardized single-trial runner callback.
"""

import copy
import itertools
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional

import numpy as np
import torch

from bokete.reporting import experiment_report, multi_trial_report
from bokete.training import Trainer, EarlyStopping, Checkpoint
from bokete.utils import (
    close_file_loggers,
    create_run_directory,
    determine_device,
    format_duration,
    get_nested_key,
    log_trial_start,
    set_nested_key,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


def _run_trials(
    config: Dict[str, Any],
    num_trials: int,
    run_fn: Callable[..., Dict[str, Any]],
    output_dir: Optional[str] = None,
    model: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Internal helper that runs a single configuration across multiple seed trials."""
    exp_label = config.get("experiment_name") if isinstance(config, dict) else getattr(config, "experiment_name", None)
    if not exp_label or not str(exp_label).strip():
        raise ValueError("Configuration must explicitly define 'experiment_name'.")

    if output_dir is None:
        run_dir_path = create_run_directory(config, attach_file_logger=True)
        output_dir = str(run_dir_path)
    else:
        setup_logging(log_file=Path(output_dir) / "experiment.log")

    setup_logging()
    all_metrics = []

    start_dt = datetime.now()
    start_time_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.time()

    try:
        base_seed = config.get("seed", 42) if isinstance(config, dict) else getattr(config, "seed", 42)
        for i in range(1, num_trials + 1):
            trial_seed = base_seed + (i - 1)
            log_trial_start(i, num_trials, exp_label, seed=trial_seed)
            try:
                metrics = run_fn(i, config, output_dir=output_dir)
            except TypeError:
                try:
                    metrics = run_fn(i, config)
                except TypeError:
                    metrics = run_fn(config)
            if metrics:
                all_metrics.append(metrics)
                val_losses = metrics.get('val_loss') or []
                if val_losses:
                    best_val = min(val_losses)
                    logger.info(f"[BOKeTE] Trial {i} complete — Best Val Loss: {best_val:.4f}")
                else:
                    logger.info(f"[BOKeTE] Trial {i} complete")
    except KeyboardInterrupt:
        logger.warning("")
        logger.warning("[BOKeTE] Multi-trial run cancelled by user (KeyboardInterrupt). Finalizing completed trials...")
        logger.warning("")
    finally:
        close_file_loggers()

    t1 = time.time()
    end_dt = datetime.now()
    end_time_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
    total_duration = round(t1 - t0, 2)

    if output_dir and all_metrics:
        multi_trial_report(
            config,
            all_metrics,
            save_path=output_dir,
            start_time=start_time_str,
            end_time=end_time_str,
            duration_seconds=total_duration,
            model=model,
        )

    # Log summary statistics if validation losses were recorded
    best_val_losses = [min(m['val_loss']) for m in all_metrics if m and 'val_loss' in m and m['val_loss']]
    if best_val_losses:
        mean_val = float(np.mean(best_val_losses))
        std_val = float(np.std(best_val_losses))
        min_val = float(np.min(best_val_losses))
        max_val = float(np.max(best_val_losses))
        summary_label = f" for [{exp_label}]" if exp_label else ""
        logger.info("")
        logger.info(
            f"[BOKeTE] === Experiment Complete{summary_label} ({len(best_val_losses)} Trials) | "
            f"Time Taken: {format_duration(total_duration)} (Start: {start_time_str}, End: {end_time_str}) ==="
        )
        logger.info(f"[BOKeTE] Mean Best Val Loss: {mean_val:.4f} ± {std_val:.4f} (Min: {min_val:.4f}, Max: {max_val:.4f})")
        logger.info("")

    return all_metrics


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

    grid = param_grid
    if grid is None and isinstance(target_config, dict):
        grid = target_config.get("param_grid")
    elif grid is None and is_dataclass(target_config):
        grid = getattr(target_config, "param_grid", None)

    if grid:
        keys = list(grid.keys())
        value_combinations = list(itertools.product(*grid.values()))
        results = []
        total = len(value_combinations)

        base_cfg = copy.deepcopy(target_config)
        if isinstance(base_cfg, dict):
            base_cfg.pop("param_grid", None)

        n_trials = num_trials or (base_cfg.get("trials", 1) if isinstance(base_cfg, dict) else getattr(base_cfg, "trials", 1))

        for idx, combo in enumerate(value_combinations, 1):
            run_config = copy.deepcopy(base_cfg)
            params_used = {}
            for key, value in zip(keys, combo):
                set_nested_key(run_config, key, value)
                params_used[key] = value

            params_str = ", ".join([f"{k}={v}" for k, v in params_used.items()])
            log_trial_start(idx, total, params_str)

            trial_metrics = _run_trials(
                config=run_config,
                num_trials=n_trials,
                run_fn=run_fn,
                output_dir=output_dir,
                model=model,
            )
            results.append({
                "parameters": params_used,
                "metrics": trial_metrics,
            })
        return results

    n_trials = num_trials or (target_config.get("trials", 1) if isinstance(target_config, dict) else getattr(target_config, "trials", 1))
    return _run_trials(
        config=target_config,
        num_trials=n_trials,
        run_fn=run_fn,
        output_dir=output_dir,
        model=model,
    )


def create_trial_runner(
    model_factory: Callable[[Dict[str, Any]], Any],
    loader_factory: Callable[[Dict[str, Any]], Any],
    criterion_factory: Optional[Callable[[Dict[str, Any]], Any]] = None,
    optimizer_factory: Optional[Callable[[Any, Dict[str, Any]], Any]] = None,
    test_fn: Optional[Callable[[Any, Dict[str, Any]], Dict[str, Any]]] = None,
    log_model_structure: bool = True,
) -> Callable[..., Dict[str, Any]]:
    """Helper factory that constructs a standardized single-trial runner callback for bokete.run_experiments.

    Args:
        model_factory: Callable `(config) -> torch.nn.Module`.
        loader_factory: Callable `(config) -> (train_loader, val_loader)`.
        criterion_factory: Optional callable `(config) -> criterion`.
        optimizer_factory: Optional callable `(model, config) -> optimizer`.
        test_fn: Optional callable `(model, config) -> dict` returning extra test metrics (e.g. Test Accuracy).
        log_model_structure: Whether to log the full PyTorch model layer structure on trial 1 (default: True).

    Returns:
        A callback `(trial_num, config, output_dir=None) -> metrics_dict` compatible with bokete.run_experiments.
    """
    def run_fn(trial_num: int, config: Dict[str, Any], output_dir: Optional[str] = None) -> Dict[str, Any]:
        exp_name = config.get("experiment_name") if isinstance(config, dict) else getattr(config, "experiment_name", None)
        if not exp_name or not str(exp_name).strip():
            raise ValueError("Configuration must explicitly define 'experiment_name'.")

        # 1. Set seed per trial for stochastic variance
        base_seed = config.get("seed", 42)
        set_seed(base_seed + (trial_num - 1))

        # 2. Build dataloaders & model
        train_loader, val_loader, *test = loader_factory(config)
        test_loader = test[0] if test else None

        model = model_factory(config)

        # 3. Build criterion & optimizer
        if criterion_factory:
            criterion = criterion_factory(config)
        else:
            loss_name = get_nested_key(config, "training.loss") or "CrossEntropyLoss"
            criterion = getattr(torch.nn, loss_name)()

        if optimizer_factory:
            optimizer = optimizer_factory(model, config)
        else:
            opt_name = get_nested_key(config, "training.optimizer") or "Adam"
            lr = get_nested_key(config, "training.lr") or 1e-3
            opt_cls = getattr(torch.optim, opt_name)
            optimizer = opt_cls(model.parameters(), lr=lr)

        # 4. Determine device & run Trainer (log device and structure on trial 1)
        is_first_trial = (trial_num == 1)
        device = determine_device(verbose=is_first_trial)
        should_log_struct = log_model_structure and is_first_trial

        trainer = Trainer(
            model,
            criterion,
            optimizer,
            device=device,
            log_model_structure=should_log_struct,
        )

        epochs = get_nested_key(config, "training.epochs") or 5
        patience = get_nested_key(config, "training.patience")
        early_stopping_obj = EarlyStopping(patience=patience) if patience is not None and patience > 0 else None
        save_ckpts = get_nested_key(config, "training.save_checkpoints")
        if save_ckpts is None:
            save_ckpts = True

        if not output_dir:
            output_dir = str(create_run_directory(config, attach_file_logger=False))

        trial_dir = Path(output_dir) / f"trial_{trial_num}"

        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            early_stopping=early_stopping_obj,
            checkpoint=Checkpoint(trial_dir / "checkpoints", enabled=save_ckpts),
        )

        # 5. Evaluate post-training test metrics if test_fn provided
        extra_metrics = None
        if test_fn:
            extra_metrics = test_fn(model, test_loader, config) if test_loader is not None else test_fn(model, config)

        # 6. Generate per-trial Markdown report
        experiment_report(
            config=config,
            metrics=history,
            model=model,
            extra_metrics=extra_metrics,
            save_path=trial_dir,
            title=f"{exp_name} - Trial {trial_num}",
        )

        result = history.as_dict()
        if extra_metrics:
            result["extra_metrics"] = extra_metrics
        return result

    return run_fn
