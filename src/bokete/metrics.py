"""
Metric tracking for training runs: the History record and summary statistics
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrainingMetrics:
    """Per-epoch record of a training run, returned by Trainer.fit()."""

    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    best_epoch: int | None = None       # 1-indexed epoch with the lowest validation loss
    best_val_loss: float | None = None
    stopped_early: bool = False

    @property
    def epochs_run(self):
        return len(self.train_loss)

    def as_dict(self):
        """Return the plain {'train_loss': [...], 'val_loss': [...]} shape."""
        return {'train_loss': self.train_loss, 'val_loss': self.val_loss}


def training_report(metrics):
    """
    Compute summary statistics for a completed training run.

    Accepts a TrainingMetrics instance or any mapping with 'train_loss'/'val_loss' lists,
    and returns a dict of final/mean losses, the best epoch and epochs run.
    """
    if isinstance(metrics, dict):
        train_loss, val_loss = metrics['train_loss'], metrics['val_loss']
    else:
        train_loss, val_loss = metrics.train_loss, metrics.val_loss

    if not train_loss or not val_loss:
        raise ValueError("Cannot summarise empty metrics (no epochs recorded).")

    return {
        'final_train_loss': train_loss[-1],
        'mean_train_loss': float(np.mean(train_loss)),
        'final_val_loss': val_loss[-1],
        'mean_val_loss': float(np.mean(val_loss)),
        'best_epoch': int(np.argmin(val_loss)) + 1,
        'best_val_loss': min(val_loss),
        'epochs_run': len(train_loss),
    }
