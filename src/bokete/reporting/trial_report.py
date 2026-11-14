"""Markdown report generation for a single training trial."""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import bokete.metrics as metrics_mod
import bokete.plotting as plotting
import bokete.utils as utils

from ._shared import (
    _format_config_table,
    _format_model_section,
    _resolve_device_string,
    _runtime_section,
    _write_report,
)


def trial_report(
    config: Dict[str, Any],
    metrics_summary: Optional[Dict[str, Any]] = None,
    train_loss: Optional[List[float]] = None,
    val_loss: Optional[List[float]] = None,
    graph_filename: Optional[str] = "graph.png",
    extra_metrics: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
    metrics: Optional[Union[metrics_mod.TrainingMetrics, Dict[str, Any]]] = None,
    auto_plot: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    model: Optional[Any] = None,
) -> str:
    """Generates a structured GFM Markdown report string for a single trial run."""
    out_file = utils.resolve_output_path(save_path, "report.md")
    config_dict, param_rows = _format_config_table(config)
    configured_monitor = utils.get_nested_key(config_dict, "training.monitor")
    graph_available = True

    # Detect if a TrainingMetrics or metrics dict was passed as 2nd positional argument
    if hasattr(metrics_summary, 'train_loss') or (
        isinstance(metrics_summary, dict) and 'train_loss' in metrics_summary and 'final_train_loss' not in metrics_summary
    ):
        metrics = metrics_summary
        metrics_summary = None

    if metrics is not None:
        m = metrics if isinstance(metrics, dict) else metrics.as_dict()
        selection = metrics_mod.selection_summary(m, configured_monitor)
        m = {**m, **selection}

        if metrics_summary is None:
            metrics_summary = metrics_mod.training_report(m) if m.get('train_loss') else dict(m)
        else:
            metrics_mod.selection_summary(metrics_summary, selection['monitor'])
            metrics_summary = {**metrics_summary, **selection}

        if train_loss is None:
            train_loss = m.get('train_loss', [])
        if val_loss is None:
            val_loss = m.get('val_loss', [])

        if auto_plot and graph_filename and m.get('train_loss') and m.get('val_loss'):
            graph_path = (out_file.parent / graph_filename) if out_file else Path(graph_filename)
            graph_available = plotting.plot_loss_curves(
                metrics=m, best_epoch=selection['best_epoch'], path=graph_path,
                title=title or "Training and Validation Loss",
            )

    metrics_summary = dict(metrics_summary or {})
    monitor = configured_monitor or metrics_summary.get('monitor', 'val_loss')
    metrics_mod.selection_summary(metrics_summary, monitor)
    metrics_summary['monitor'] = monitor
    train_loss = train_loss or []
    val_loss = val_loss or []

    exp_name = config_dict.get('experiment_name')
    if title:
        header_title = title
    elif exp_name:
        header_title = f"Experiment Report: {exp_name}"
    else:
        header_title = "Experiment Report"

    # Build Trial Overview dynamically
    overview_bullets = []
    recorded = m if metrics is not None else metrics_summary
    status = "interrupted" if recorded.get("interrupted") else "completed"
    overview_bullets.append(f"- **Status:** `{status}`")
    if recorded.get("best_checkpoint_available") is False:
        overview_bullets.append("- **Best Checkpoint:** unavailable; best-weight test metrics were not evaluated.")

    # 1. Experiment Name
    if exp_name and exp_name != config_dict.get('dataset'):
        overview_bullets.append(f"- **Experiment Name:** `{exp_name}`")

    # 2. Dataset Configuration
    dataset_val = config_dict.get('dataset_name') or config_dict.get('dataset')
    if dataset_val:
        overview_bullets.append(f"- **Dataset Configuration:** `{dataset_val}`")

    # 3. Execution Timing
    start_time = metrics_summary.get('start_time')
    end_time = metrics_summary.get('end_time')
    duration_sec = metrics_summary.get('duration_seconds')
    duration_fmt = metrics_summary.get('duration_formatted') or utils.format_duration(duration_sec)

    if start_time:
        overview_bullets.append(f"- **Start Date:** `{start_time}`")
    else:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        overview_bullets.append(f"- **Execution Date:** `{now_str}`")

    if end_time:
        overview_bullets.append(f"- **End Date:** `{end_time}`")

    if duration_sec is not None:
        overview_bullets.append(f"- **Time Taken:** `{duration_fmt}` (`{duration_sec:.2f}s`)")

    # 4. Key Hyperparameters
    flat_config = utils.flatten_dict(config_dict)
    lr = flat_config.get('lr') or flat_config.get('training.lr')
    batch_size = flat_config.get('batch_size') or flat_config.get('training.batch_size')
    optimizer = flat_config.get('optimizer') or flat_config.get('training.optimizer')

    hyperparams_list = []
    if lr is not None:
        hyperparams_list.append(f"`lr={lr}`")
    if batch_size is not None:
        hyperparams_list.append(f"`batch_size={batch_size}`")
    if optimizer is not None:
        hyperparams_list.append(f"`optimizer={optimizer}`")

    if hyperparams_list:
        overview_bullets.append(f"- **Key Hyperparameters:** {', '.join(hyperparams_list)}")

    # 5. Device
    device_name = metrics_summary.get('device_name')
    if not device_name and hasattr(metrics, 'device_name'):
        device_name = metrics.device_name
    device_str = _resolve_device_string(config_dict, device_name)
    overview_bullets.append(f"- **Execution Device:** `{device_str}`")

    # 6. Total Epochs & Best Epoch
    if train_loss:
        overview_bullets.append(f"- **Total Epochs Run:** `{len(train_loss)}`")

    best_epoch = metrics_summary.get('best_epoch')
    monitor = metrics_summary.get('monitor', 'val_loss')
    mode = metrics_mod.monitor_mode(monitor)
    overview_bullets.append(f"- **Selection Metric:** `{monitor}` ({'minimise' if mode == 'min' else 'maximise'})")
    best_score = metrics_summary.get('best_score')
    if best_score is None and monitor == 'val_loss':
        best_score = metrics_summary.get('best_val_loss')
    if best_epoch and best_score is not None:
        overview_bullets.append(f"- **Selected Epoch:** `{best_epoch}` ({utils.format_metric_name(monitor)}: `{best_score}`)")
    elif best_epoch:
        overview_bullets.append(f"- **Selected Epoch:** `{best_epoch}`")

    # 7. Model Architecture Metadata
    model_bullet, model_section = _format_model_section(model)
    if model_bullet:
        overview_bullets.append(model_bullet)

    overview_section = "\n".join(overview_bullets) if overview_bullets else "- No overview metadata recorded."

    # Build Custom/Extra Metrics table rows if present
    extra_rows = ""
    if extra_metrics:
        extra_rows = "\n" + "\n".join([f"| {utils.format_metric_name(k)} | `{v}` |" for k, v in extra_metrics.items()])
    for key, label in (("selected_val_accuracy", "Selected Epoch Validation Accuracy"),
                       ("final_val_accuracy", "Final Validation Accuracy"),
                       ("selected_val_loss", "Selected Epoch Validation Loss")):
        if metrics_summary.get(key) is not None:
            extra_rows += f"\n| {label} | `{metrics_summary[key]}` |"

    # Build Collapsible Per-Epoch Details Table
    accuracy = m.get('val_accuracy', []) if metrics is not None else []
    accuracy_header = " Validation Accuracy |" if accuracy else ""
    accuracy_align = " :---: |" if accuracy else ""
    epoch_rows = "\n".join([
        f"| {epoch + 1} | {t:.4f} | {v:.4f} |" + (f" {accuracy[epoch]:.4f} |" if accuracy else "")
        for epoch, (t, v) in enumerate(zip(train_loss, val_loss))
    ])

    loss_graph = (f"\n## Loss Curves\n![Loss Curve](./{graph_filename})\n"
                  if graph_available and graph_filename and train_loss and val_loss else "")

    runtime_section = _runtime_section([recorded])

    report_md = f"""# {header_title}

## Trial Overview
{overview_section}
{model_section}
## Performance Metrics
| Metric | Value |
| :--- | :--- |
| Final Train Loss | `{metrics_summary.get('final_train_loss', 'N/A')}` |
| Final Validation Loss | `{metrics_summary.get('final_val_loss', 'N/A')}` |
| Mean Train Loss | `{metrics_summary.get('mean_train_loss', 'N/A')}` |
| Mean Validation Loss | `{metrics_summary.get('mean_val_loss', 'N/A')}` |{extra_rows}

{loss_graph}
{runtime_section}

## Hyperparameters & Configuration
| Parameter | Value |
| :--- | :--- |
{param_rows}

<details>
<summary><b>View Full Epoch Breakdown ({len(train_loss)} Epochs)</b></summary>

| Epoch | Train Loss | Validation Loss |{accuracy_header}
| :---: | :---: | :---: |{accuracy_align}
{epoch_rows}

</details>
"""

    _write_report(out_file, report_md)
    return report_md
