from datetime import datetime
from typing import Dict, Any, List, Optional


def experiment_report(
    config: Dict[str, Any],
    metrics_summary: Dict[str, Any],
    train_loss: List[float],
    val_loss: List[float],
    graph_filename: str = "graph.png",
    extra_metrics: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
) -> str:
    """Generates a structured GFM Markdown report string for an experiment run."""
    flat_config = _flatten_dict(config)
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

    return f"""# {header_title}

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


def _flatten_dict(d, parent_key=''):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def multi_trial_report(
    config: Dict[str, Any],
    all_trial_metrics: List[Dict[str, Any]]
) -> str:
    """Generates a structured GFM Markdown report summarizing a multi-trial experiment."""
    import numpy as np

    best_val_losses = [min(m['val_loss']) for m in all_trial_metrics if m and 'val_loss' in m and m['val_loss']]
    if not best_val_losses:
        return "# Multi-Trial Experiment Summary\n\nNo trial metrics recorded."

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
    flat_config = _flatten_dict(config)
    exp_name = config.get('experiment_name') or config.get('exp_name') or config.get('name')
    header_title = f"Multi-Trial Experiment Summary: {exp_name}" if (exp_name and exp_name != config.get('dataset')) else "Multi-Trial Experiment Summary"

    exp_bullet = f"- **Experiment Name:** `{exp_name}`\n" if (exp_name and exp_name != config.get('dataset')) else ""

    return f"""# {header_title}

## Aggregate Performance ({len(all_trial_metrics)} Trials)
{exp_bullet}- **Mean Best Validation Loss:** `{mean_val:.4f} ± {std_val:.4f}`
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
