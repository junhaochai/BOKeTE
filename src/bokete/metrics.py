"""
Metric tracking for training runs: the History record and summary statistics
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

    @property
    def epochs_run(self) -> int:
        return len(self.train_loss)

    def as_dict(self) -> Dict[str, list[float]]:
        """Return the plain {'train_loss': [...], 'val_loss': [...]} shape."""
        return {'train_loss': self.train_loss, 'val_loss': self.val_loss}


def training_report(metrics: Union[TrainingMetrics, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary statistics for a completed training run.

    Accepts a TrainingMetrics instance or any mapping with 'train_loss'/'val_loss' lists,
    and returns a dict of final/mean losses, the best epoch and epochs run.
    """
    if isinstance(metrics, dict):
        train_loss = metrics.get('train_loss', [])
        val_loss = metrics.get('val_loss', [])
    else:
        train_loss = metrics.train_loss
        val_loss = metrics.val_loss

    if not train_loss:
        raise ValueError("Cannot summarise empty metrics (no training epochs recorded).")

    summary: Dict[str, Any] = {
        'final_train_loss': train_loss[-1],
        'mean_train_loss': float(np.mean(train_loss)),
        'epochs_run': len(train_loss),
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
