"""
Orchestration of multiple experiments, multi-trial runs, and parameter sweeps.
"""

import copy
import itertools
import logging
import os
from typing import Dict, Any, List, Callable, Optional

import numpy as np

from bokete.reporting import multi_trial_report
from bokete.utils import set_nested_key, log_trial_start

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
    all_metrics = []
    dataset_name = config.get('dataset_name') or config.get('dataset') or ""

    try:
        for i in range(1, num_trials + 1):
            log_trial_start(i, num_trials, dataset_name)
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

    if output_dir and all_metrics:
        report_md = multi_trial_report(config, all_metrics)
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
        exp_label = f" for [{dataset_name}]" if dataset_name else ""
        logger.info("")
        logger.info(f"[BOKeTE] === Experiment Complete{exp_label} ({len(best_val_losses)} Trials) ===")
        logger.info(f"[BOKeTE] Mean Best Val Loss: {mean_val:.4f} ± {std_val:.4f} (Min: {min_val:.4f}, Max: {max_val:.4f})")
        logger.info("")

    return all_metrics
