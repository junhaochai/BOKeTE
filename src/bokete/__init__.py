"""bokete: simple PyTorch helpers for model training and experimentation"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bokete.metrics import TrainingMetrics, training_report
    from bokete.plotting import plot_loss_curves
    from bokete.training import (
        Checkpoint,
        EarlyStopping,
        Trainer,
        TrainingInterrupted,
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
        format_metric_name,
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
    from bokete.reporting import (
        config_report,
        experiment_report,
        trial_report,
    )

__version__ = "0.2.0.dev0"

__all__ = [
    "BOKeTEFormatter",
    "Colours",
    "TqdmLoggingHandler",
    "Checkpoint",
    "EarlyStopping",
    "TrainingMetrics",
    "TrainingInterrupted",
    "close_file_loggers",
    "config_report",
    "create_run_directory",
    "create_trial_runner",
    "determine_device",
    "evaluate",
    "experiment_report",
    "flatten_dict",
    "format_duration",
    "format_metric_name",
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
    "plot_loss_curves",
    "set_nested_key",
    "set_seed",
    "setup_logging",
    "training_report",
    "trial_report",
    "Trainer",
    "run_experiments",
]

_EXPORT_MODULES = {
    "BOKeTEFormatter": "utils",
    "Colours": "colours",
    "TqdmLoggingHandler": "utils",
    "Checkpoint": "training",
    "EarlyStopping": "training",
    "TrainingMetrics": "metrics",
    "TrainingInterrupted": "training",
    "close_file_loggers": "utils",
    "config_report": "reporting",
    "create_run_directory": "utils",
    "create_trial_runner": "experiments",
    "determine_device": "utils",
    "evaluate": "training",
    "experiment_report": "reporting",
    "flatten_dict": "utils",
    "format_duration": "utils",
    "format_metric_name": "utils",
    "get_device_name": "utils",
    "get_nested_key": "utils",
    "load_config": "utils",
    "load_config_as": "utils",
    "parse_cli_config": "utils",
    "resolve_config_path": "utils",
    "log_dataset_info": "utils",
    "log_experiment_start": "utils",
    "log_model_info": "utils",
    "log_trial_start": "utils",
    "plot_loss_curves": "plotting",
    "set_nested_key": "utils",
    "set_seed": "utils",
    "setup_logging": "utils",
    "training_report": "metrics",
    "trial_report": "reporting",
    "Trainer": "training",
    "run_experiments": "experiments",
}


def __getattr__(name):
    """Load public objects when requested, keeping package and CLI imports light."""
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
