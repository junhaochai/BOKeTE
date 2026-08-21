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
    Colours,
    TqdmLoggingHandler,



    close_file_loggers,
    create_run_directory,
    determine_device,
    flatten_dict,
    format_duration,
    get_device_name,
    get_nested_key,
    load_config,
    load_config_as,
    parse_cli_config,
    resolve_config_path,
    log_dataset_info,
    log_experiment_start,
    log_model_info,
    log_trial_start,
    set_nested_key,
    set_seed,
    setup_logging,
)
from bokete.experiments import (
    create_trial_runner,
    run_experiments,
)
from bokete.reporting import experiment_report, multi_trial_report

__version__ = "0.1.1"

__all__ = [
    "BOKeTEFormatter",
    "Colours",
    "TqdmLoggingHandler",
    "Checkpoint",
    "EarlyStopping",
    "TrainingMetrics",
    "close_file_loggers",
    "create_run_directory",
    "create_trial_runner",
    "determine_device",
    "evaluate",
    "experiment_report",
    "flatten_dict",
    "format_duration",
    "get_device_name",
    "get_nested_key",
    "load_config",
    "load_config_as",
    "parse_cli_config",
    "resolve_config_path",
    "log_dataset_info",
    "log_experiment_start",
    "log_model_info",
    "log_trial_start",
    "multi_trial_report",
    "plot_loss_curves",
    "set_nested_key",
    "set_seed",
    "setup_logging",
    "training_report",
    "Trainer",
    "run_experiments",
]
