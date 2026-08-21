"""
Markdown report generation for single experiments and multi-trial runs.

Key Functions:
  - experiment_report : Generates a structured GFM Markdown report string and saves to file.
  - multi_trial_report: Aggregates statistics across multiple experimental trial runs.
"""

from dataclasses import is_dataclass
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from bokete.metrics import TrainingMetrics, training_report
from bokete.plotting import plot_loss_curves
from bokete.utils import flatten_dict

logger = logging.getLogger(__name__)


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
    save_path: Optional[Union[str, Path]] = None,
    model: Optional[Any] = None,
) -> str:
    """Generates a structured GFM Markdown report string for an experiment run."""
    # Resolve target output file and parent directory (handles directories and Path objects)
    out_file: Optional[Path] = None
    if save_path:
        p = Path(save_path)
        out_file = (p / "report.md") if (p.is_dir() or p.suffix == "" or not p.name.endswith(".md")) else p

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
            graph_path = (out_file.parent / graph_filename) if out_file else Path(graph_filename)
            plot_loss_curves(metrics=metrics, path=graph_path, title=title or "Training and Validation Loss")

    metrics_summary = metrics_summary or {}
    train_loss = train_loss or []
    val_loss = val_loss or []

    if is_dataclass(config):
        from dataclasses import asdict
        config = asdict(config)

    flat_config = flatten_dict(config)
    param_rows = "\n".join([f"| `{k}` | `{v}` |" for k, v in flat_config.items()])

    exp_name = config.get('experiment_name')
    if title:
        header_title = title
    elif exp_name:
        header_title = f"Experiment Report: {exp_name}"
    else:
        header_title = "Experiment Report"

    # Build Trial Overview dynamically so it works across any PyTorch project
    overview_bullets = []

    # 1. Experiment Name (if present in config)
    if exp_name and exp_name != config.get('dataset'):
        overview_bullets.append(f"- **Experiment Name:** `{exp_name}`")

    # 2. Dataset Configuration
    dataset_val = config.get('dataset_name') or config.get('dataset')
    if dataset_val:
        overview_bullets.append(f"- **Dataset Configuration:** `{dataset_val}`")

    # 3. Execution Start, End Date & Time Taken
    from bokete.utils import format_duration, get_device_name, determine_device
    import torch

    start_time = metrics_summary.get('start_time')
    end_time = metrics_summary.get('end_time')
    duration_sec = metrics_summary.get('duration_seconds')
    duration_fmt = metrics_summary.get('duration_formatted') or format_duration(duration_sec)

    if start_time:
        overview_bullets.append(f"- **Start Date:** `{start_time}`")
    else:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        overview_bullets.append(f"- **Execution Date:** `{now_str}`")

    if end_time:
        overview_bullets.append(f"- **End Date:** `{end_time}`")

    if duration_sec is not None:
        overview_bullets.append(f"- **Time Taken:** `{duration_fmt}` (`{duration_sec:.2f}s`)")

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

    # 5. Device (with GPU model detection)
    device_name = metrics_summary.get('device_name')
    if not device_name and hasattr(metrics, 'device_name'):
        device_name = metrics.device_name

    if device_name:
        overview_bullets.append(f"- **Execution Device:** `{device_name}`")
    else:
        dev_cfg = config.get('device')
        dev_obj = torch.device(dev_cfg) if dev_cfg else determine_device(verbose=False)
        dev_gpu = get_device_name(dev_obj)
        dev_str = f"{dev_obj} ({dev_gpu})" if (dev_gpu and dev_gpu.lower() != dev_obj.type.lower()) else str(dev_obj)
        overview_bullets.append(f"- **Execution Device:** `{dev_str}`")

    # 6. Total Epochs & Best Epoch
    if train_loss:
        overview_bullets.append(f"- **Total Epochs Run:** `{len(train_loss)}`")

    best_epoch = metrics_summary.get('best_epoch')
    best_val_loss = metrics_summary.get('best_val_loss')
    if best_epoch and best_val_loss:
        overview_bullets.append(f"- **Best Epoch:** `{best_epoch}` (Validation Loss: `{best_val_loss}`)")
    elif best_epoch:
        overview_bullets.append(f"- **Best Epoch:** `{best_epoch}`")

    # 7. Model Architecture Metadata
    model_section = ""
    if model is not None:
        if hasattr(model, "parameters"):
            raw_m = getattr(model, "module", model)
            m_name = raw_m.__class__.__name__
            tot_p = sum(p.numel() for p in raw_m.parameters())
            trn_p = sum(p.numel() for p in raw_m.parameters() if p.requires_grad)
            struct_str = str(raw_m)
        elif isinstance(model, dict):
            m_name = model.get("model_name", "PyTorch Model")
            tot_p = model.get("total_params", 0)
            trn_p = model.get("trainable_params", 0)
            struct_str = model.get("structure", "")
        else:
            m_name = getattr(model, "__class__", type(model)).__name__
            tot_p = 0
            trn_p = 0
            struct_str = str(model)

        overview_bullets.append(
            f"- **Model Architecture:** `{m_name}` (`{tot_p:,}` total params | `{trn_p:,}` trainable)"
        )

        layer_block = ""
        if struct_str:
            layer_block = (
                f"\n<details>\n<summary><b>View Model Layer Hierarchy</b></summary>\n\n"
                f"```text\n{struct_str}\n```\n\n</details>\n"
            )

        model_section = (
            f"\n## Model Architecture\n"
            f"- **Model Class:** `{m_name}`\n"
            f"- **Total Parameters:** `{tot_p:,}`\n"
            f"- **Trainable Parameters:** `{trn_p:,}`\n"
            f"{layer_block}"
        )

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
{model_section}
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
    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(report_md, encoding="utf-8")
        logger.info(f"[BOKeTE] Saved Experiment Report to: {out_file}")

    return report_md


def multi_trial_report(
    config: Dict[str, Any],
    all_trial_metrics: List[Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    duration_seconds: Optional[float] = None,
) -> str:
    """Generates a structured GFM Markdown report summarizing a multi-trial experiment."""
    import numpy as np
    import torch
    from bokete.utils import format_duration, get_device_name, determine_device

    out_file: Optional[Path] = None
    if save_path:
        p = Path(save_path)
        out_file = (p / "multi_trial_report.md") if (p.is_dir() or p.suffix == "" or not p.name.endswith(".md")) else p

    best_val_losses = [min(m['val_loss']) for m in all_trial_metrics if m and 'val_loss' in m and m['val_loss']]
    if not best_val_losses:
        report_md = "# Multi-Trial Experiment Summary\n\nNo trial metrics recorded."
        if out_file:
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

    start_line = f"- **Start Date:** `{start_time}`\n" if start_time else ""
    end_line = f"- **End Date:** `{end_time}`\n" if end_time else ""
    dur_line = f"- **Total Time Taken:** `{format_duration(duration_seconds)}` (`{duration_seconds:.2f}s`)\n" if duration_seconds is not None else ""

    dev_cfg = config.get('device')
    dev_obj = torch.device(dev_cfg) if dev_cfg else determine_device(verbose=False)
    dev_gpu = get_device_name(dev_obj)
    dev_str = f"{dev_obj} ({dev_gpu})" if (dev_gpu and dev_gpu.lower() != dev_obj.type.lower()) else str(dev_obj)
    dev_line = f"- **Execution Device:** `{dev_str}`\n"

    if not start_time:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_line = f"- **Execution Date:** `{now_str}`\n"

    report_md = f"""# {header_title}

## Aggregate Performance ({len(all_trial_metrics)} Trials)
{exp_bullet}{start_line}{end_line}{dur_line}{dev_line}- **Mean Best Validation Loss:** `{mean_val:.4f} ± {std_val:.4f}`
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
    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(report_md, encoding="utf-8")
        logger.info(f"[BOKeTE] Saved Multi-Trial Report to: {out_file}")

    return report_md
