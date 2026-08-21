"""
Orchestration of multiple experiments and parameter sweeps.
"""

import copy
import itertools
import logging
from typing import Dict, Any, List, Callable

logger = logging.getLogger(__name__)


def _set_nested_key(d: Dict[str, Any], key_path: str, value: Any) -> None:
    """Sets a value in a nested dictionary using a dot-notation path (e.g., 'training.lr')."""
    parts = key_path.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


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
    
    for combo in value_combinations:
        # Deep copy to prevent mutations from bleeding into other runs
        run_config = copy.deepcopy(base_config)
        
        # Apply the current parameter combination
        params_used = {}
        for key, value in zip(keys, combo):
            _set_nested_key(run_config, key, value)
            params_used[key] = value
            
        logger.info(f"=== Starting Sweep Configuration: {params_used} ===")
        metrics = run_fn(run_config)
        
        results.append({
            "parameters": params_used,
            "metrics": metrics
        })
        
    return results
