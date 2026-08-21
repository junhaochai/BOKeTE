"""
General utility functions for configuration I/O, directory management, hardware detection, and logging.

Key Classes & Functions:
  - load_config         : Parses YAML configuration files into dictionaries.
  - load_config_as      : Loads YAML files directly into user-defined @dataclass schemas.
  - parse_cli_config    : Parses CLI arguments for --config and loads YAML directly into @dataclass schemas.
  - create_run_directory: Creates timestamped output subfolders (results/exp_name/YYYYMMDD-XX/).
  - set_seed            : Sets random seeds across Python, NumPy, and PyTorch for reproducibility.
  - determine_device    : Auto-detects optimal hardware accelerator (CUDA / MPS / CPU).
  - log_dataset_info    : Formats and logs standardized dataset sample counts.
  - log_trial_start     : Emits standardized trial section headers to console/log.
  - BOKeTEFormatter     : Custom logging formatter that handles clean blank lines.
  - flatten_dict        : Flattens nested dictionary keys into dot-notation strings.
  - get_nested_key      : Safely retrieves a value from a nested dictionary by key path.
  - set_nested_key      : Updates a value deep inside a nested dictionary by key path.
"""

import argparse
from dataclasses import fields, is_dataclass
from datetime import datetime
import logging
import os
import random
from pathlib import Path
from typing import Dict, Any, Optional, Sequence, Type, TypeVar

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
    """Loads a YAML configuration file into a dictionary and logs the event."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    if path.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError(
            f"Unsupported configuration file extension '{path.suffix}'. Only .yaml and .yml files are supported."
        )

    logger.info("")
    logger.info(f"[BOKeTE] Loading configuration from: {path}")

    with open(path, "r", encoding="utf-8") as f:
        import yaml

        return yaml.safe_load(f) or {}


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
    base_dir: str | Path | Any = "results",
    experiment_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    attach_file_logger: bool = True,
    config: Optional[Any] = None,
) -> Path:
    """Creates a timestamped experiment directory (e.g. results/exp_name/YYYYMMDD-01)
    and optionally attaches a file logger (experiment.log) to the root logger.

    Accepts a config dict/dataclass directly: `create_run_directory(cfg)`
    """
    if is_dataclass(base_dir) or isinstance(base_dir, dict) or hasattr(base_dir, "experiment_name"):
        config = base_dir
        base_dir = "results"

    if config is not None:
        if base_dir == "results":
            if hasattr(config, "output_dir") and getattr(config, "output_dir"):
                base_dir = getattr(config, "output_dir")
            elif hasattr(config, "base_dir") and getattr(config, "base_dir"):
                base_dir = getattr(config, "base_dir")
            elif isinstance(config, dict):
                base_dir = config.get("output_dir") or config.get("base_dir") or "results"

        if experiment_name is None:
            if hasattr(config, "experiment_name"):
                experiment_name = getattr(config, "experiment_name")
            elif isinstance(config, dict):
                experiment_name = config.get("experiment_name") or config.get("exp_name")
        if dataset_name is None:
            if hasattr(config, "dataset_name"):
                dataset_name = getattr(config, "dataset_name")
            elif hasattr(config, "dataset") and hasattr(config.dataset, "name"):
                dataset_name = getattr(config.dataset, "name")
            elif isinstance(config, dict):
                d = config.get("dataset")
                if isinstance(d, dict):
                    dataset_name = d.get("name")
                elif isinstance(d, str):
                    dataset_name = d

    base_path = Path(base_dir)
    now = datetime.now()
    date_prefix = now.strftime("%Y%m%d")

    subfolder = experiment_name or dataset_name or "experiment"
    target_parent = base_path / subfolder
    target_parent.mkdir(parents=True, exist_ok=True)

    existing = [
        d for d in target_parent.iterdir()
        if d.is_dir() and d.name.startswith(date_prefix)
    ]
    run_dir = target_parent / f"{date_prefix}-{len(existing)+1:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if attach_file_logger:
        log_path = run_dir / "experiment.log"
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


def get_nested_key(d: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Safely retrieves a value from a nested dictionary using a dot-notation path (e.g., 'training.lr').

    Handles missing keys, non-dict intermediate values, and explicit None entries without raising exceptions.
    """
    keys = key_path.split(".")
    curr: Any = d
    for k in keys:
        if not isinstance(curr, dict) or k not in curr:
            return default
        curr = curr[k]
    return curr if curr is not None else default


def set_nested_key(d: Dict[str, Any], key_path: str, value: Any) -> None:
    """Sets a value in a nested dictionary using a dot-notation path (e.g., 'training.lr')."""
    parts = key_path.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


T = TypeVar("T")


def load_config_as(config_path: str | Path, dataclass_cls: Type[T]) -> T:
    """Loads a YAML configuration file directly into a user-defined @dataclass instance.

    Recursively converts nested dictionaries into nested dataclass instances where applicable.
    Unspecified fields automatically retain their dataclass default values.
    """
    raw = load_config(config_path)

    def _from_dict(cls: Type[Any], data: Dict[str, Any]) -> Any:
        if not is_dataclass(cls) or not isinstance(data, dict):
            return data

        field_types = {f.name: f.type for f in fields(cls)}
        kwargs = {}
        for f in fields(cls):
            if f.name in data and data[f.name] is not None:
                val = data[f.name]
                target_type = field_types[f.name]
                if is_dataclass(target_type) and isinstance(val, dict):
                    kwargs[f.name] = _from_dict(target_type, val)
                else:
                    kwargs[f.name] = val
        return cls(**kwargs)

    return _from_dict(dataclass_cls, raw or {})


def parse_cli_config(
    dataclass_cls: Type[T],
    default_config: str | Path | None = None,
    description: str = "PyTorch Training Pipeline via BOKeTE",
    args_list: Optional[Sequence[str]] = None,
) -> T:
    """Parses command-line arguments for a --config YAML file path and loads it into a dataclass schema."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(default_config) if default_config else None,
        help="Path to YAML configuration file",
        required=default_config is None,
    )
    parsed_args = parser.parse_args(args_list)
    return load_config_as(parsed_args.config, dataclass_cls)
