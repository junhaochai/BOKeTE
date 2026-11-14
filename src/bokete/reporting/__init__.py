"""
Markdown report generation for single trials, configurations, and experiment sweeps.

Key Functions:
  - trial_report: Generates a structured GFM Markdown report string for a single trial.
  - config_report: Aggregates statistics across multiple experimental trial runs for a configuration.
  - experiment_report: Aggregates parameter combinations into a master comparative leaderboard.
"""

from .config_report import config_report
from .experiment_report import experiment_report
from .trial_report import trial_report


__all__ = ["config_report", "experiment_report", "trial_report"]
