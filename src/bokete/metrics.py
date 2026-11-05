"""
Metric tracking and summary statistics for training runs.

Key Classes & Functions:
  - TrainingMetrics: Dataclass recording per-epoch loss history and run status.
  - training_report: Computes summary statistics (best epoch, final loss, delta) from metrics.
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Dict, Any, Union, Optional

import numpy as np

import bokete.utils as utils


@dataclass
class TrainingMetrics:
    """Per-epoch record of a training run, returned by Trainer.fit()."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: Optional[int] = None       # 1-indexed epoch selected by monitor/mode
    best_val_loss: Optional[float] = None  # Minimum loss, independent of selected epoch
    stopped_early: bool = False
    interrupted: bool = False
    start_time: Optional[str] = None       # Formatted start timestamp (%Y-%m-%d %H:%M:%S)
    end_time: Optional[str] = None         # Formatted end timestamp (%Y-%m-%d %H:%M:%S)
    duration_seconds: Optional[float] = None
    device_name: Optional[str] = None
    val_accuracy: list[float] = field(default_factory=list)
    monitor: str = "val_loss"
    best_score: Optional[float] = None

    @property
    def epochs_run(self) -> int:
        return len(self.train_loss)

    def as_dict(self) -> Dict[str, Any]:
        """Return dictionary representation of metric history and run status."""
        return {k: v for k, v in asdict(self).items() if v is not None}


def monitor_mode(monitor: str) -> str:
    """Infer the direction for a supported validation metric."""
    directions = {"val_loss": "min", "val_accuracy": "max"}
    if not isinstance(monitor, str) or monitor not in directions:
        raise ValueError("monitor must be 'val_loss' or 'val_accuracy'.")
    return directions[monitor]


def is_improvement(score: float, best: Optional[float], mode: str, min_delta: float = 0.0) -> bool:
    """Compare finite scores; ties retain the earlier epoch."""
    if not math.isfinite(score):
        raise ValueError("The monitored validation score must be finite.")
    return best is None or (score < best - min_delta if mode == "min" else score > best + min_delta)


def selection_summary(m: Dict[str, Any], monitor: Optional[str] = None) -> Dict[str, Any]:
    """Select an epoch from recorded values, including histories from older runs."""
    monitor = monitor or m.get("monitor", "val_loss")
    mode = monitor_mode(monitor)
    if m.get("monitor", monitor) != monitor or m.get("mode", mode) != mode:
        raise ValueError("Cannot compare runs with different monitoring settings.")
    values = m.get(monitor, [])
    if monitor == "val_accuracy" and not values and (m.get("train_loss") or m.get("val_loss")):
        raise ValueError("Validation accuracy was selected but no accuracy history was recorded.")
    if values and m.get("val_loss") and len(values) != len(m["val_loss"]):
        raise ValueError("Monitored metric and validation loss histories must have equal lengths.")
    best_score = None
    best_epoch = None
    for epoch, score in enumerate(values, 1):
        if is_improvement(score, best_score, mode):
            best_score, best_epoch = float(score), epoch
    return {"monitor": monitor, "mode": mode, "best_score": best_score, "best_epoch": best_epoch}


def training_report(metrics: Union[TrainingMetrics, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary statistics for a completed training run.

    Accepts a TrainingMetrics instance or any mapping with 'train_loss'/'val_loss' lists,
    and returns a dict of final/mean losses, timing information, best epoch and epochs run.
    """
    m = metrics if isinstance(metrics, dict) else metrics.as_dict()
    train_loss = m.get('train_loss', [])
    val_loss = m.get('val_loss', [])
    start_time = m.get('start_time')
    end_time = m.get('end_time')
    duration_seconds = m.get('duration_seconds')
    device_name = m.get('device_name')

    if not train_loss:
        raise ValueError("Cannot summarise empty metrics (no training epochs recorded).")

    has_val = bool(val_loss)
    summary: Dict[str, Any] = {
        'final_train_loss': train_loss[-1],
        'mean_train_loss': float(np.mean(train_loss)),
        'final_val_loss': val_loss[-1] if has_val else None,
        'mean_val_loss': float(np.mean(val_loss)) if has_val else None,
        'best_val_loss': float(np.min(val_loss)) if has_val else None,
        'epochs_run': len(train_loss),
        'start_time': start_time,
        'end_time': end_time,
        'duration_seconds': duration_seconds,
        'duration_formatted': utils.format_duration(duration_seconds),
        'device_name': device_name,
    }

    summary.update(selection_summary(m))
    selected_epoch = summary['best_epoch']
    summary['selected_val_loss'] = val_loss[selected_epoch - 1] if selected_epoch and val_loss else None
    accuracy = m.get('val_accuracy', [])
    if accuracy:
        summary['final_val_accuracy'] = accuracy[-1]
        summary['selected_val_accuracy'] = accuracy[selected_epoch - 1] if selected_epoch else None

    return summary
