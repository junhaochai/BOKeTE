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
from typing import Dict, Any, List, Callable, Optional

import numpy as np

from bokete.reporting import multi_trial_report
from bokete.utils import set_nested_key, log_trial_start, setup_logging, format_duration

logger = logging.getLogger(__name__)


def run_experiments(
    base_config: Dict[str, Any],
    param_grid: Dict[str, List[Any]],
    run_fn: Callable[[Dict[str, Any]], Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Generates all combinations of parameters from param_grid, updates base_config,
    and executes run_fn for each run.

    Args:
        base_config: The default configuration dictionary.
        param_grid: Dict mapping config paths (e.g., 'training.lr') to lists of values to test.
        run_fn: A callback function `(config) -> metrics_dict` that runs a single experiment.

    Returns:
        A list of results, each containing the parameters used and the returned metrics.
    """
    keys = list(param_grid.keys())
    value_combinations = list(itertools.product(*param_grid.values()))

    results = []

    total = len(value_combinations)
    for idx, combo in enumerate(value_combinations, 1):
        # Deep copy to prevent mutations from bleeding into other runs
        run_config = copy.deepcopy(base_config)

        # Apply the current parameter combination
        params_used = {}
        for key, value in zip(keys, combo):
            set_nested_key(run_config, key, value)
            params_used[key] = value

        params_str = ", ".join([f"{k}={v}" for k, v in params_used.items()])
        log_trial_start(idx, total, params_str)
        metrics = run_fn(run_config)

        results.append({
            "parameters": params_used,
            "metrics": metrics
        })

    return results



def run_trials(
    config: Dict[str, Any],
    num_trials: int,
    run_fn: Callable[[int, Dict[str, Any]], Dict[str, Any]],
    output_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Runs a single configuration across multiple trials with standardized logging,
    graceful Ctrl+C cancellation, and optional multi-trial Markdown report generation.
    """
    setup_logging()
    all_metrics = []
    exp_label = (
        config.get('experiment_name')
        or config.get('dataset_name')
        or config.get('dataset')
        or "Trial Execution"
    )

    start_dt = datetime.now()
    start_time_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.time()

    try:
        for i in range(1, num_trials + 1):
            log_trial_start(i, num_trials, exp_label)
            metrics = run_fn(i, config)
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

    t1 = time.time()
    end_dt = datetime.now()
    end_time_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
    total_duration = round(t1 - t0, 2)

    if output_dir and all_metrics:
        report_md = multi_trial_report(
            config,
            all_metrics,
            save_path=output_dir,
            start_time=start_time_str,
            end_time=end_time_str,
            duration_seconds=total_duration,
        )
        report_path = os.path.join(output_dir, "summary-report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        logger.info(f"[BOKeTE] Saved Multi-Trial Summary Report to: {report_path}")

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


def create_trial_runner(
    model_factory: Callable[[Dict[str, Any]], Any],
    loader_factory: Callable[[Dict[str, Any]], Any],
    criterion_factory: Optional[Callable[[Dict[str, Any]], Any]] = None,
    optimizer_factory: Optional[Callable[[Any, Dict[str, Any]], Any]] = None,
    log_model_structure: bool = True,
) -> Callable[[int, Dict[str, Any]], Dict[str, Any]]:
    """Helper factory that constructs a standardized single-trial runner callback for bokete.run_trials.

    Args:
        model_factory: Callable `(config) -> torch.nn.Module`.
        loader_factory: Callable `(config) -> (train_loader, val_loader)`.
        criterion_factory: Optional callable `(config) -> criterion`.
        optimizer_factory: Optional callable `(model, config) -> optimizer`.
        log_model_structure: Whether to log the full PyTorch model layer structure on trial 1 (default: True).

    Returns:
        A callback `(trial_num, config) -> metrics_dict` compatible with bokete.run_trials.
    """
    import torch
    from pathlib import Path
    from bokete.training import Trainer, EarlyStopping, Checkpoint
    from bokete.reporting import experiment_report
    from bokete.utils import set_seed, determine_device, get_nested_key

    def run_fn(trial_num: int, config: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Set seed per trial for stochastic variance
        base_seed = config.get("seed", 42)
        set_seed(base_seed + (trial_num - 1))

        # 2. Build dataloaders & model
        train_loader, val_loader = loader_factory(config)
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
        verbose_device = (trial_num == 1)
        device = determine_device(verbose=verbose_device)
        should_log_struct = (log_model_structure and trial_num == 1)

        trainer = Trainer(
            model,
            criterion,
            optimizer,
            device=device,
            log_model_structure=should_log_struct,
        )

        epochs = get_nested_key(config, "training.epochs") or 5
        patience = get_nested_key(config, "training.patience") or 3
        save_ckpts = get_nested_key(config, "training.save_checkpoints")
        if save_ckpts is None:
            save_ckpts = True

        output_base = config.get("output_dir") or "results"
        exp_name = config.get("experiment_name") or "experiment"
        trial_dir = Path(output_base) / exp_name / f"trial_{trial_num}"

        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            early_stopping=EarlyStopping(patience=patience),
            checkpoint=Checkpoint(trial_dir / "checkpoints", enabled=save_ckpts),
        )

        # 5. Generate per-trial Markdown report
        experiment_report(
            config=config,
            metrics=history,
            model=model,
            save_path=trial_dir,
            title=f"{exp_name} - Trial {trial_num}",
        )

        if hasattr(history, "as_dict"):
            return history.as_dict()
        elif hasattr(history, "to_dict"):
            return history.to_dict()
        elif is_dataclass(history):
            from dataclasses import asdict
            return asdict(history)
        return history

    return run_fn


# Convenient shorthand alias
trial_runner = create_trial_runner
