"""
bokete:  simple PyTorch helpers for model training and experimentation
"""

from bokete.metrics import TrainingMetrics, training_report
from bokete.plotting import plot_loss_curves
from bokete.training import (
    Checkpoint,
    EarlyStopping,
    Trainer,
    determine_device,
    evaluate,
    set_seed,
)
from bokete.experiments import run_experiments
from bokete.reporting import experiment_report, multi_trial_report

__version__ = "0.1.0"

__all__ = [
    "Checkpoint",
    "EarlyStopping",
    "TrainingMetrics",
    "determine_device",
    "evaluate",
    "experiment_report",
    "multi_trial_report",
    "plot_loss_curves",
    "set_seed",
    "training_report",
    "Trainer",
    "run_experiments",
]
