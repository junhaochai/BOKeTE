"""Markdown summaries across trials of one configuration."""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import numpy as np

import bokete.metrics as metrics_mod
import bokete.plotting as plotting
import bokete.utils as utils

from ._shared import (
    _extra_metric_keys,
    _extra_metric_stats,
    _format_config_table,
    _format_model_section,
    _resolve_device_string,
    _runtime_section,
    _write_report,
)


def config_report(
    config: Dict[str, Any],
    all_trial_metrics: List[Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    model: Optional[Any] = None,
    graph_filename: Optional[str] = "multi_trial_loss.png",
    auto_plot: bool = True,
) -> str:
    """Aggregates statistics across multiple experimental trial runs for a configuration (config-report.md)."""
    out_file = utils.resolve_output_path(save_path, "config-report.md")
    config_dict, param_rows = _format_config_table(config)
    default_monitor = all_trial_metrics[0].get('monitor', 'val_loss') if all_trial_metrics else 'val_loss'
    monitor = utils.get_nested_key(config_dict, "training.monitor", default=default_monitor)
    mode = metrics_mod.monitor_mode(monitor)
    metric_label = utils.format_metric_name(monitor)
    selections = [metrics_mod.selection_summary(m, monitor) for m in all_trial_metrics]
    completed_metrics = [m for m in all_trial_metrics if not m.get("interrupted")]

    graph_available = False
    if auto_plot and graph_filename and out_file and completed_metrics:
        graph_path = out_file.parent / graph_filename
        graph_available = plotting.plot_multi_trial_loss_curves(
            all_trial_metrics=completed_metrics,
            path=graph_path,
            title="Multi-Trial Training & Validation Loss (Mean ± Std)",
        )

    scored_trials = [(all_trial_metrics[idx - 1].get('trial', idx), s['best_score'])
                     for idx, s in enumerate(selections, 1)
                     if s['best_score'] is not None and not all_trial_metrics[idx - 1].get('interrupted')]

    best_val_losses = [score for _, score in scored_trials]
    if best_val_losses:
        mean_val = float(np.mean(best_val_losses))
        std_val = float(np.std(best_val_losses))
        min_val = float(np.min(best_val_losses))
        max_val = float(np.max(best_val_losses))
        best_index = int(np.argmin(best_val_losses) if mode == 'min' else np.argmax(best_val_losses))
        best_trial_idx = scored_trials[best_index][0]
        best_value = best_val_losses[best_index]
        aggregate_scores = (
            f"- **Mean Best {metric_label}:** `{mean_val:.4f} ± {std_val:.4f}`\n"
            f"- **Best Trial {metric_label}:** `{best_value:.4f}` (Trial {best_trial_idx})\n"
            f"- **Lowest Selected {metric_label}:** `{min_val:.4f}`\n"
            f"- **Highest Selected {metric_label}:** `{max_val:.4f}`\n"
        )
    else:
        aggregate_scores = "No completed trial scores recorded.\n"

    # Extract unique extra_metrics keys across all trials
    extra_keys = _extra_metric_keys(all_trial_metrics)
    extra_stats = _extra_metric_stats(completed_metrics, extra_keys)

    extra_summary_lines = []
    for k, (m_val, s_val) in extra_stats.items():
        fmt_k = utils.format_metric_name(k)
        # Prevent duplicate prefix if metric is already aggregated (e.g. 'Mean Accuracy')
        label = fmt_k if fmt_k.lower().startswith(("mean ", "average ")) else f"Mean {fmt_k}"
        extra_summary_lines.append(f"- **{label}:** `{m_val:.4f} ± {s_val:.4f}`\n")

    extra_summary_str = "".join(extra_summary_lines)

    trial_rows = []
    for idx, m in enumerate(all_trial_metrics, 1):
        v_loss = m.get('val_loss', []) if isinstance(m, dict) else []
        t_loss = m.get('train_loss', []) if isinstance(m, dict) else []
        b_val = selections[idx - 1]['best_score']
        b_val = b_val if b_val is not None else 'N/A'
        b_ep = selections[idx - 1]['best_epoch'] or 'N/A'
        f_tr = t_loss[-1] if t_loss else 'N/A'
        f_vl = v_loss[-1] if v_loss else 'N/A'

        row_cells = [f"| Trial {m.get('trial', idx)}", f"`{b_ep}`", f"`{b_val}`", f"`{f_tr}`", f"`{f_vl}`"]
        if extra_keys:
            em = m.get('extra_metrics', {}) if isinstance(m, dict) else {}
            for ek in extra_keys:
                row_cells.append(f"`{em.get(ek, 'N/A')}`")
        row_cells.append(f"`{'interrupted' if m.get('interrupted') else 'completed'}`")
        trial_rows.append(" | ".join(row_cells) + " |")

    trial_table_header = f"| Trial | Selected Epoch | Selected {metric_label} | Final Train Loss | Final Validation Loss |"
    trial_table_align = "| :---: | :---: | :---: | :---: | :---: |"
    if extra_keys:
        formatted_extra_headers = [utils.format_metric_name(ek) for ek in extra_keys]
        trial_table_header += " " + " | ".join(formatted_extra_headers) + " |"
        trial_table_align += " " + " | ".join([":---:"] * len(extra_keys)) + " |"
    trial_table_header += " Status |"
    trial_table_align += " :---: |"

    trial_table = f"{trial_table_header}\n{trial_table_align}\n" + "\n".join(trial_rows)

    exp_name = config_dict.get('experiment_name') or config_dict.get('exp_name') or config_dict.get('name')
    header_title = f"Configuration Report: {exp_name}" if (exp_name and exp_name != config_dict.get('dataset')) else "Configuration Report"

    exp_bullet = f"- **Experiment Name:** `{exp_name}`\n" if (exp_name and exp_name != config_dict.get('dataset')) else ""

    start_line = f"- **Start Date:** `{start_time}`\n" if start_time else f"- **Execution Date:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n"
    end_line = f"- **End Date:** `{end_time}`\n" if end_time else ""
    dur_line = f"- **Total Time Taken:** `{utils.format_duration(duration_seconds)}` (`{duration_seconds:.2f}s`)\n" if duration_seconds is not None else ""

    dev_str = _resolve_device_string(config_dict)
    dev_line = f"- **Execution Device:** `{dev_str}`\n"

    model_bullet, model_section = _format_model_section(model)
    model_line = f"{model_bullet}\n" if model_bullet else ""

    loss_graph_section = ""
    if graph_available:
        loss_graph_section = f"\n## Multi-Trial Loss Curves\n![Multi-Trial Loss Curve](./{graph_filename})\n"

    runtime_section = _runtime_section(all_trial_metrics)
    status = "interrupted" if any(m.get('interrupted') for m in all_trial_metrics) else "completed"
    report_md = f"""# {header_title}

## Aggregate Performance ({len(all_trial_metrics)} Trials)
{exp_bullet}{start_line}{end_line}{dur_line}{dev_line}{model_line}- **Selection Metric:** `{monitor}` ({'minimise' if mode == 'min' else 'maximise'})
- **Status:** `{status}`
- **Completed Trials:** `{len(completed_metrics)}`; interrupted trials are excluded from aggregates.
{aggregate_scores}
{extra_summary_str}{model_section}
## Per-Trial Performance Breakdown
{trial_table}
{loss_graph_section}
{runtime_section}
## Experiment Configuration
| Parameter | Value |
| :--- | :--- |
{param_rows}
"""

    _write_report(out_file, report_md)
    return report_md
