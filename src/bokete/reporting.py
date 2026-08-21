"""
Markdown report generation for single experiments and multi-trial runs.

Key Functions:
  - experiment_report : Generates a structured GFM Markdown report string and saves to file.
  - multi_trial_report: Aggregates statistics across multiple experimental trial runs.
"""

from dataclasses import is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from bokete.metrics import TrainingMetrics, training_report
from bokete.plotting import plot_loss_curves
from bokete.utils import flatten_dict


def experiment_report(
    config: Dict[str, Any],
    metrics_summary: Optional[Dict[str, Any]] = None,
    train_loss: Optional[List[float]] = None,
    val_loss: Optional[List[float]] = None,
    graph_filename: Optional[str] = "graph.png",
    extra_metrics: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
    metrics: Optional[Union[TrainingMetrics, Dict[str, Any]]] = None,
    auto_plot: bool = True,
    save_path: Optional[str] = None,
) -> str:
    """Generates a structured GFM Markdown report string for an experiment run."""
    # Detect if a TrainingMetrics or metrics dict was passed as 2nd positional argument
    if hasattr(metrics_summary, 'train_loss') or (
        isinstance(metrics_summary, dict) and 'train_loss' in metrics_summary and 'final_train_loss' not in metrics_summary
    ):
        metrics = metrics_summary
        metrics_summary = None

    if metrics is not None:
        if metrics_summary is None:
            metrics_summary = training_report(metrics)
        if train_loss is None:
            if hasattr(metrics, 'train_loss'):
                train_loss = metrics.train_loss
            elif isinstance(metrics, dict):
                train_loss = metrics.get('train_loss', [])
        if val_loss is None:
            if hasattr(metrics, 'val_loss'):
                val_loss = metrics.val_loss
            elif isinstance(metrics, dict):
                val_loss = metrics.get('val_loss', [])

        if auto_plot and graph_filename:
            plot_loss_curves(metrics=metrics, path=graph_filename, title=title or "Training and Validation Loss")

    metrics_summary = metrics_summary or {}
    train_loss = train_loss or []
    val_loss = val_loss or []

    if is_dataclass(config):
        from dataclasses import asdict
        config = asdict(config)

    flat_config = flatten_dict(config)
    param_rows = "\n".join([f"| `{k}` | `{v}` |" for k, v in flat_config.items()])

    exp_name = config.get('experiment_name') or config.get('exp_name') or config.get('name')
    if title:
        header_title = title
    elif exp_name and exp_name != config.get('dataset'):
        header_title = f"Experiment Trial Report: {exp_name}"
    else:
        header_title = "Experiment Trial Report"

    # Build Trial Overview dynamically so it works across any PyTorch project
    overview_bullets = []

    # 1. Experiment Name (if present in config)
    if exp_name and exp_name != config.get('dataset'):
        overview_bullets.append(f"- **Experiment Name:** `{exp_name}`")

    # 2. Dataset Configuration
    dataset_val = config.get('dataset_name') or config.get('dataset')
    if dataset_val:
        overview_bullets.append(f"- **Dataset Configuration:** `{dataset_val}`")

    # 3. Execution Date & Time
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    overview_bullets.append(f"- **Execution Date:** `{now_str}`")

    # 4. Key Hyperparameters (extracted dynamically)
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
    device_val = config.get('device')
    if device_val:
        overview_bullets.append(f"- **Execution Device:** `{device_val}`")

    # 6. Total Epochs & Best Epoch
    if train_loss:
        overview_bullets.append(f"- **Total Epochs Run:** `{len(train_loss)}`")

    best_epoch = metrics_summary.get('best_epoch')
    best_val_loss = metrics_summary.get('best_val_loss')
    if best_epoch and best_val_loss:
        overview_bullets.append(f"- **Best Epoch:** `{best_epoch}` (Validation Loss: `{best_val_loss}`)")
    elif best_epoch:
        overview_bullets.append(f"- **Best Epoch:** `{best_epoch}`")

    overview_section = "\n".join(overview_bullets) if overview_bullets else "- No overview metadata recorded."

    # Build Custom/Extra Metrics table rows if present
    extra_rows = ""
    if extra_metrics:
        extra_rows = "\n" + "\n".join([f"| {k} | `{v}` |" for k, v in extra_metrics.items()])

    # Build Collapsible Per-Epoch Details Table
    epoch_rows = "\n".join([
        f"| {epoch + 1} | {t:.4f} | {v:.4f} |"
        for epoch, (t, v) in enumerate(zip(train_loss, val_loss))
    ])

    report_md = f"""# {header_title}

## Trial Overview
{overview_section}

## Performance Metrics
| Metric | Value |
| :--- | :--- |
| Final Train Loss | `{metrics_summary.get('final_train_loss', 'N/A')}` |
| Final Validation Loss | `{metrics_summary.get('final_val_loss', 'N/A')}` |
| Mean Train Loss | `{metrics_summary.get('mean_train_loss', 'N/A')}` |
| Mean Validation Loss | `{metrics_summary.get('mean_val_loss', 'N/A')}` |{extra_rows}

## Loss Curves
![Loss Curve](./{graph_filename})

## Hyperparameters & Configuration
| Parameter | Value |
| :--- | :--- |
{param_rows}

<details>
<summary><b>View Full Epoch Breakdown ({len(train_loss)} Epochs)</b></summary>

| Epoch | Train Loss | Validation Loss |
| :---: | :---: | :---: |
{epoch_rows}

</details>
"""
    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(report_md, encoding="utf-8")

    return report_md


def multi_trial_report(
    config: Dict[str, Any],
    all_trial_metrics: List[Dict[str, Any]],
    save_path: Optional[str] = None,
) -> str:
    """Generates a structured GFM Markdown report summarizing a multi-trial experiment."""
    import numpy as np

    best_val_losses = [min(m['val_loss']) for m in all_trial_metrics if m and 'val_loss' in m and m['val_loss']]
    if not best_val_losses:
        report_md = "# Multi-Trial Experiment Summary\n\nNo trial metrics recorded."
        if save_path:
            out_file = Path(save_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(report_md, encoding="utf-8")
        return report_md

    mean_val = float(np.mean(best_val_losses))
    std_val = float(np.std(best_val_losses))
    min_val = float(np.min(best_val_losses))
    max_val = float(np.max(best_val_losses))
    best_trial_idx = int(np.argmin(best_val_losses)) + 1

    trial_rows = []
    for idx, m in enumerate(all_trial_metrics, 1):
        v_loss = m.get('val_loss', [])
        t_loss = m.get('train_loss', [])
        b_val = min(v_loss) if v_loss else 'N/A'
        b_ep = int(np.argmin(v_loss)) + 1 if v_loss else 'N/A'
        f_tr = t_loss[-1] if t_loss else 'N/A'
        f_vl = v_loss[-1] if v_loss else 'N/A'
        trial_rows.append(f"| Trial {idx} | `{b_ep}` | `{b_val}` | `{f_tr}` | `{f_vl}` |")

    trial_table = "\n".join(trial_rows)
    flat_config = flatten_dict(config)
    param_rows = "\n".join([f"| `{k}` | `{v}` |" for k, v in flat_config.items()])
    exp_name = config.get('experiment_name') or config.get('exp_name') or config.get('name')
    header_title = f"Multi-Trial Experiment Summary: {exp_name}" if (exp_name and exp_name != config.get('dataset')) else "Multi-Trial Experiment Summary"

    exp_bullet = f"- **Experiment Name:** `{exp_name}`\n" if (exp_name and exp_name != config.get('dataset')) else ""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_md = f"""# {header_title}

## Aggregate Performance ({len(all_trial_metrics)} Trials)
{exp_bullet}- **Execution Date:** `{now_str}`
- **Mean Best Validation Loss:** `{mean_val:.4f} ± {std_val:.4f}`
- **Lowest Validation Loss (Best Trial):** `{min_val:.4f}` (Trial {best_trial_idx})
- **Highest Validation Loss:** `{max_val:.4f}`

## Per-Trial Performance Breakdown
| Trial | Best Epoch | Best Val Loss | Final Train Loss | Final Validation Loss |
| :---: | :---: | :---: | :---: | :---: |
{trial_table}

## Experiment Configuration
| Parameter | Value |
| :--- | :--- |
{param_rows}
"""
    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(report_md, encoding="utf-8")

    return report_md
