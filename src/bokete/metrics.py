"""
Metric tracking and summary statistics for training runs.

Key Classes & Functions:
  - TrainingMetrics: Dataclass recording per-epoch loss history and run status.
  - training_report: Computes summary statistics (best epoch, final loss, delta) from metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Union, Optional

import numpy as np


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
        """Return metric history dictionary including timing and device parameters."""
        res: Dict[str, Any] = {'train_loss': self.train_loss, 'val_loss': self.val_loss}
        if self.start_time:
            res['start_time'] = self.start_time
        if self.end_time:
            res['end_time'] = self.end_time
        if self.duration_seconds is not None:
            res['duration_seconds'] = self.duration_seconds
        if self.device_name:
            res['device_name'] = self.device_name
        return res


def training_report(metrics: Union[TrainingMetrics, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary statistics for a completed training run.

    Accepts a TrainingMetrics instance or any mapping with 'train_loss'/'val_loss' lists,
    and returns a dict of final/mean losses, timing information, best epoch and epochs run.
    """
    if isinstance(metrics, dict):
        train_loss = metrics.get('train_loss', [])
        val_loss = metrics.get('val_loss', [])
        start_time = metrics.get('start_time')
        end_time = metrics.get('end_time')
        duration_seconds = metrics.get('duration_seconds')
        device_name = metrics.get('device_name')
    else:
        train_loss = metrics.train_loss
        val_loss = metrics.val_loss
        start_time = metrics.start_time
        end_time = metrics.end_time
        duration_seconds = metrics.duration_seconds
        device_name = metrics.device_name

    if not train_loss:
        raise ValueError("Cannot summarise empty metrics (no training epochs recorded).")

    from bokete.utils import format_duration

    summary: Dict[str, Any] = {
        'final_train_loss': train_loss[-1],
        'mean_train_loss': float(np.mean(train_loss)),
        'epochs_run': len(train_loss),
        'start_time': start_time,
        'end_time': end_time,
        'duration_seconds': duration_seconds,
        'duration_formatted': format_duration(duration_seconds) if duration_seconds is not None else None,
        'device_name': device_name,
    }

    if val_loss:
        summary.update({
            'final_val_loss': val_loss[-1],
            'mean_val_loss': float(np.mean(val_loss)),
            'best_epoch': int(np.argmin(val_loss)) + 1,
            'best_val_loss': float(np.min(val_loss)),
        })
    else:
        summary.update({
            'final_val_loss': None,
            'mean_val_loss': None,
            'best_epoch': None,
            'best_val_loss': None,
        })

    return summary
