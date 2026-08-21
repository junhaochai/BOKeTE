"""
General utility functions for configuration I/O, directory management, logging,
reproducibility, hardware detection, and dictionary manipulation.
"""

from datetime import datetime
import logging
import os
import random
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class BOKeTEFormatter(logging.Formatter):
    """Custom logging formatter that emits clean unformatted blank lines when message is empty."""
    def format(self, record: logging.LogRecord) -> str:
        if not record.msg or record.msg == "\n":
            return ""
        return super().format(record)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU and all CUDA devices) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def determine_device(verbose: bool = True) -> torch.device:
    """Return the best available PyTorch device: CUDA, MPS (Mac), or CPU."""
    if torch.cuda.is_available():
        dev = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        dev = torch.device('mps')
    else:
        dev = torch.device('cpu')

    if verbose:
        logger.info(f"[BOKeTE] Using device: {dev}")

    return dev


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Loads a YAML or JSON configuration file into a dictionary and logs the event."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    logger.info("")
    logger.info(f"[BOKeTE] Loading configuration from: {path}")

    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml
            return yaml.safe_load(f) or {}
        elif path.suffix.lower() == ".json":
            import json
            return json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file extension: {path.suffix}")


def log_dataset_info(
    name: str,
    train_samples: Optional[int] = None,
    val_samples: Optional[int] = None,
    **details: Any,
) -> None:
    """Logs standardized dataset configuration and sample counts for any PyTorch dataset."""
    detail_str = ""
    if details:
        items = [f"{k.replace('_', ' ').title()}: {v}" for k, v in details.items()]
        detail_str = f" ({', '.join(items)})"
    logger.info(f"[BOKeTE] Dataset configuration: {name}{detail_str}")
    if train_samples is not None:
        logger.info(f"[BOKeTE] Loaded Train dataset subset: {train_samples:,} samples")
    if val_samples is not None:
        logger.info(f"[BOKeTE] Loaded Validation dataset subset: {val_samples:,} samples")


def log_trial_start(trial_num: int, total_trials: int, name: str = "") -> None:
    """Logs a standardized trial header with clean unformatted console spacing."""
    logger.info("")
    label = f" [{name}]" if name else ""
    logger.info(f"[BOKeTE] --- Starting Trial {trial_num}/{total_trials}{label} ---")


def create_run_directory(
    base_dir: str = "results",
    experiment_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    attach_file_logger: bool = True,
) -> str:
    """Creates a timestamped experiment directory (e.g. results/exp_name/YYYYMMDD-01)
    and optionally attaches a file logger (experiment.log) to the root logger.
    """
    now = datetime.now()
    date_prefix = now.strftime("%Y%m%d")

    subfolder = experiment_name or dataset_name or "experiment"
    target_parent = os.path.join(base_dir, subfolder)
    os.makedirs(target_parent, exist_ok=True)

    existing = [
        d for d in os.listdir(target_parent)
        if os.path.isdir(os.path.join(target_parent, d)) and d.startswith(date_prefix)
    ]
    run_dir = os.path.join(target_parent, f"{date_prefix}-{len(existing)+1:02d}")
    os.makedirs(run_dir, exist_ok=True)

    if attach_file_logger:
        log_path = os.path.join(run_dir, "experiment.log")
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        formatter = BOKeTEFormatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        for h in root_logger.handlers:
            h.setFormatter(formatter)

    logger.info(f"[BOKeTE] Created Experiment Run Directory: {run_dir}")
    return run_dir


def flatten_dict(d: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
    """Flattens a nested dictionary using dot notation (e.g. {'training': {'lr': 0.001}} -> {'training.lr': 0.001})."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def set_nested_key(d: Dict[str, Any], key_path: str, value: Any) -> None:
    """Sets a value in a nested dictionary using a dot-notation path (e.g., 'training.lr')."""
    parts = key_path.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value
