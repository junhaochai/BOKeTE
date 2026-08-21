"""
Metric tracking and summary statistics for training runs.

Key Classes & Functions:
  - TrainingMetrics: Dataclass recording per-epoch loss history and run status.
  - training_report: Computes summary statistics (best epoch, final loss, delta) from metrics.
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, Any, Union, Optional

import numpy as np

import bokete.utils as utils


@dataclass
class TrainingMetrics:
    """Per-epoch record of a training run, returned by Trainer.fit()."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: Optional[int] = None       # 1-indexed epoch with the lowest validation loss
    best_val_loss: Optional[float] = None
    stopped_early: bool = False
    interrupted: bool = False
    start_time: Optional[str] = None       # Formatted start timestamp (%Y-%m-%d %H:%M:%S)
    end_time: Optional[str] = None         # Formatted end timestamp (%Y-%m-%d %H:%M:%S)
    duration_seconds: Optional[float] = None
    device_name: Optional[str] = None

    @property
    def epochs_run(self) -> int:
        return len(self.train_loss)

    def as_dict(self) -> Dict[str, Any]:
        """Return dictionary representation of metric history and run status."""
        return {k: v for k, v in asdict(self).items() if v is not None}


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
        'best_epoch': int(np.argmin(val_loss)) + 1 if has_val else None,
        'best_val_loss': float(np.min(val_loss)) if has_val else None,
        'epochs_run': len(train_loss),
        'start_time': start_time,
        'end_time': end_time,
        'duration_seconds': duration_seconds,
        'duration_formatted': utils.format_duration(duration_seconds),
        'device_name': device_name,
    }

    return summary
