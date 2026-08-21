"""
bokete: simple PyTorch helpers for model training and experimentation
"""

from bokete.metrics import TrainingMetrics, training_report
from bokete.plotting import plot_loss_curves
from bokete.training import (
    Checkpoint,
    EarlyStopping,
    Trainer,
    evaluate,
)
from bokete.utils import (
    BOKeTEFormatter,
    create_run_directory,
    determine_device,
    flatten_dict,
    load_config,
    log_dataset_info,
    log_trial_start,
    set_nested_key,
    set_seed,
)
from bokete.experiments import (
    run_experiments,
    run_trials,
)
from bokete.reporting import experiment_report, multi_trial_report

__version__ = "0.1.1"

__all__ = [
    "BOKeTEFormatter",
    "Checkpoint",
    "EarlyStopping",
    "TrainingMetrics",
    "create_run_directory",
    "determine_device",
    "evaluate",
    "experiment_report",
    "flatten_dict",
    "load_config",
    "log_dataset_info",
    "log_trial_start",
    "multi_trial_report",
    "plot_loss_curves",
    "set_nested_key",
    "set_seed",
    "training_report",
    "Trainer",
    "run_experiments",
    "run_trials",
]
