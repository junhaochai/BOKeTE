"""
Markdown report generation for single trials, configurations, and experiment sweeps.

Key Functions:
  - trial_report: Generates a structured GFM Markdown report string for a single trial.
  - config_report: Aggregates statistics across multiple experimental trial runs for a configuration.
  - experiment_report: Aggregates parameter combinations into a master comparative leaderboard.
"""

from dataclasses import asdict, is_dataclass
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import numpy as np
import torch

import bokete.metrics as metrics_mod
import bokete.plotting as plotting
import bokete.utils as utils


logger = logging.getLogger(__name__)



def _write_report(out_file: Optional[Path], report_md: str) -> None:
    """Writes report string to out_file and logs completion."""
    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(report_md, encoding="utf-8")
        logger.info(f"[BOKeTE] Saved Report to: {out_file}")


def _resolve_device_string(
    config: Dict[str, Any], explicit_device: Optional[str] = None
) -> str:
    """Formats execution device string from summary metric or config."""
    if explicit_device:
        return explicit_device
    dev_cfg = config.get("device")
    dev_obj = torch.device(dev_cfg) if dev_cfg else utils.determine_device(verbose=False)
    dev_gpu = utils.get_device_name(dev_obj)
    if dev_gpu and dev_gpu.lower() != dev_obj.type.lower():
        return f"{dev_obj} ({dev_gpu})"
    return str(dev_obj)


def _format_config_table(config: Any) -> tuple[Dict[str, Any], str]:
    """Converts dataclass config to dict if needed and returns (dict_config, param_rows_markdown)."""
    cfg_dict = asdict(config) if is_dataclass(config) else (config or {})
    flat = utils.flatten_dict(cfg_dict)
    rows = "\n".join([f"| `{k}` | `{v}` |" for k, v in flat.items()])
    return cfg_dict, rows


def _extract_model_info(model: Any) -> tuple[str, int, int, str]:
    """Extracts (model_name, total_params, trainable_params, structure_str) from a model object or dict."""
    # Active PyTorch model instance (nn.Module, DataParallel, or DDP)
    if hasattr(model, "parameters"):
        raw_model = getattr(model, "module", model)
        total_params = sum(p.numel() for p in raw_model.parameters())
        trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        return raw_model.__class__.__name__, total_params, trainable_params, str(raw_model)

    # Loaded PyTorch checkpoint or model metadata dictionary
    if isinstance(model, dict):
        missing = [k for k in ("model_name", "total_params", "trainable_params") if k not in model]
        if missing:
            raise ValueError(f"Model checkpoint dictionary missing required key(s): {', '.join(missing)}")
        return str(model["model_name"]), int(model["total_params"]), int(model["trainable_params"]), str(model.get("structure", ""))

    # Invalid model input (neither a PyTorch model nor a checkpoint dict)
    raise ValueError(
        f"Invalid model object of type '{type(model).__name__}'. "
        "Expected a PyTorch nn.Module or a model metadata dictionary."
    )


def _format_model_section(model: Optional[Any]) -> tuple[str, str]:
    """Extracts model details and returns formatted (overview_bullet, markdown_section)."""
    # No model object passed to report generator
    if model is None:
        return "", ""

    model_name, total_params, trainable_params, structure_str = _extract_model_info(model)

    bullet = f"- **Model Architecture:** `{model_name}` (`{total_params:,}` total params | `{trainable_params:,}` trainable)"

    layer_block = ""
    if structure_str:
        layer_block = (
            f"\n<details>\n<summary><b>View Model Layer Hierarchy</b></summary>\n\n"
            f"```text\n{structure_str}\n```\n\n</details>\n"
        )

    section = (
        f"\n## Model Architecture\n"
        f"- **Model Class:** `{model_name}`\n"
        f"- **Total Parameters:** `{total_params:,}`\n"
        f"- **Trainable Parameters:** `{trainable_params:,}`\n"
        f"{layer_block}"
    )

    return bullet, section


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

    # Detect if a TrainingMetrics or metrics dict was passed as 2nd positional argument
    if hasattr(metrics_summary, 'train_loss') or (
        isinstance(metrics_summary, dict) and 'train_loss' in metrics_summary and 'final_train_loss' not in metrics_summary
    ):
        metrics = metrics_summary
        metrics_summary = None

    if metrics is not None:
        m = metrics if isinstance(metrics, dict) else metrics.as_dict()
        if metrics_summary is None:
            metrics_summary = metrics_mod.training_report(m)
        if train_loss is None:
            train_loss = m.get('train_loss', [])
        if val_loss is None:
            val_loss = m.get('val_loss', [])

        if auto_plot and graph_filename:
            graph_path = (out_file.parent / graph_filename) if out_file else Path(graph_filename)
            plotting.plot_loss_curves(metrics=metrics, path=graph_path, title=title or "Training and Validation Loss")

    metrics_summary = metrics_summary or {}
    train_loss = train_loss or []
    val_loss = val_loss or []

    config_dict, param_rows = _format_config_table(config)

    exp_name = config_dict.get('experiment_name')
    if title:
        header_title = title
    elif exp_name:
        header_title = f"Experiment Report: {exp_name}"
    else:
        header_title = "Experiment Report"

    # Build Trial Overview dynamically
    overview_bullets = []

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
    best_val_loss = metrics_summary.get('best_val_loss')
    if best_epoch and best_val_loss:
        overview_bullets.append(f"- **Best Epoch:** `{best_epoch}` (Validation Loss: `{best_val_loss}`)")
    elif best_epoch:
        overview_bullets.append(f"- **Best Epoch:** `{best_epoch}`")

    # 7. Model Architecture Metadata
    model_bullet, model_section = _format_model_section(model)
    if model_bullet:
        overview_bullets.append(model_bullet)

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
    _write_report(out_file, report_md)
    return report_md


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

    if auto_plot and graph_filename and out_file and all_trial_metrics:
        graph_path = out_file.parent / graph_filename
        plotting.plot_multi_trial_loss_curves(
            all_trial_metrics=all_trial_metrics,
            path=graph_path,
            title="Multi-Trial Training & Validation Loss (Mean ± Std)",
        )

    best_val_losses = [min(m['val_loss']) for m in all_trial_metrics if m and 'val_loss' in m and m['val_loss']]
    if not best_val_losses:
        report_md = "# Configuration Report\n\nNo trial metrics recorded."
        _write_report(out_file, report_md)
        return report_md

    mean_val = float(np.mean(best_val_losses))
    std_val = float(np.std(best_val_losses))
    min_val = float(np.min(best_val_losses))
    max_val = float(np.max(best_val_losses))
    best_trial_idx = int(np.argmin(best_val_losses)) + 1

    # Extract unique extra_metrics keys across all trials
    extra_keys: List[str] = []
    for m in all_trial_metrics:
        if isinstance(m, dict) and 'extra_metrics' in m and isinstance(m['extra_metrics'], dict):
            for k in m['extra_metrics'].keys():
                if k not in extra_keys:
                    extra_keys.append(k)

    extra_summary_lines = []
    for k in extra_keys:
        num_vals = []
        for m in all_trial_metrics:
            em = m.get('extra_metrics', {}) if isinstance(m, dict) else {}
            if k in em:
                try:
                    num_vals.append(float(em[k]))
                except (ValueError, TypeError):
                    pass
        if num_vals:
            m_val = float(np.mean(num_vals))
            s_val = float(np.std(num_vals))
            extra_summary_lines.append(f"- **Mean {k}:** `{m_val:.4f} ± {s_val:.4f}`\n")

    extra_summary_str = "".join(extra_summary_lines)

    trial_rows = []
    for idx, m in enumerate(all_trial_metrics, 1):
        v_loss = m.get('val_loss', []) if isinstance(m, dict) else []
        t_loss = m.get('train_loss', []) if isinstance(m, dict) else []
        b_val = min(v_loss) if v_loss else 'N/A'
        b_ep = int(np.argmin(v_loss)) + 1 if v_loss else 'N/A'
        f_tr = t_loss[-1] if t_loss else 'N/A'
        f_vl = v_loss[-1] if v_loss else 'N/A'

        row_cells = [f"| Trial {idx}", f"`{b_ep}`", f"`{b_val}`", f"`{f_tr}`", f"`{f_vl}`"]
        if extra_keys:
            em = m.get('extra_metrics', {}) if isinstance(m, dict) else {}
            for ek in extra_keys:
                row_cells.append(f"`{em.get(ek, 'N/A')}`")
        trial_rows.append(" | ".join(row_cells) + " |")

    trial_table_header = "| Trial | Best Epoch | Best Val Loss | Final Train Loss | Final Validation Loss |"
    trial_table_align = "| :---: | :---: | :---: | :---: | :---: |"
    if extra_keys:
        trial_table_header += " " + " | ".join(extra_keys) + " |"
        trial_table_align += " " + " | ".join([":---:"] * len(extra_keys)) + " |"

    trial_table = f"{trial_table_header}\n{trial_table_align}\n" + "\n".join(trial_rows)

    config_dict, param_rows = _format_config_table(config)
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
    if auto_plot and graph_filename:
        loss_graph_section = f"\n## Multi-Trial Loss Curves\n![Multi-Trial Loss Curve](./{graph_filename})\n"

    report_md = f"""# {header_title}

## Aggregate Performance ({len(all_trial_metrics)} Trials)
{exp_bullet}{start_line}{end_line}{dur_line}{dev_line}{model_line}- **Mean Best Validation Loss:** `{mean_val:.4f} ± {std_val:.4f}`
- **Lowest Validation Loss (Best Trial):** `{min_val:.4f}` (Trial {best_trial_idx})
- **Highest Validation Loss:** `{max_val:.4f}`
{extra_summary_str}{model_section}
## Per-Trial Performance Breakdown
{trial_table}
{loss_graph_section}
## Experiment Configuration
| Parameter | Value |
| :--- | :--- |
{param_rows}
"""
    _write_report(out_file, report_md)
    return report_md


def experiment_report(
    config: Dict[str, Any],
    sweep_results: List[Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    duration_seconds: Optional[float] = None,
) -> str:
    """Aggregates all parameter combinations and generates the master experiment-report.md leaderboard."""
    out_file = utils.resolve_output_path(save_path, "experiment-report.md")

    config_dict, param_rows = _format_config_table(config)
    exp_name = config_dict.get('experiment_name') or config_dict.get('exp_name') or config_dict.get('name')
    header_title = f"Experiment Report: {exp_name}" if exp_name else "Experiment Report"

    # Extract aggregated statistics per combo
    processed_combos = []
    all_extra_keys: List[str] = []
    for res in sweep_results:
        metrics_list = res.get("metrics", [])
        for m in metrics_list:
            if isinstance(m, dict) and "extra_metrics" in m and isinstance(m["extra_metrics"], dict):
                for k in m["extra_metrics"].keys():
                    if k not in all_extra_keys:
                        all_extra_keys.append(k)

    for idx, res in enumerate(sweep_results, 1):
        params = res.get("parameters", {})
        metrics_list = res.get("metrics", [])
        run_dir = res.get("run_dir")

        best_val_losses = [min(m['val_loss']) for m in metrics_list if m and 'val_loss' in m and m['val_loss']]
        mean_best_val = float(np.mean(best_val_losses)) if best_val_losses else None
        std_best_val = float(np.std(best_val_losses)) if best_val_losses else 0.0

        final_train_losses = [m['train_loss'][-1] for m in metrics_list if m and 'train_loss' in m and m['train_loss']]
        mean_final_train = float(np.mean(final_train_losses)) if final_train_losses else None

        final_val_losses = [m['val_loss'][-1] for m in metrics_list if m and 'val_loss' in m and m['val_loss']]
        mean_final_val = float(np.mean(final_val_losses)) if final_val_losses else None

        extra_stats = {}
        for ek in all_extra_keys:
            num_vals = []
            for m in metrics_list:
                em = m.get('extra_metrics', {}) if isinstance(m, dict) else {}
                if ek in em:
                    try:
                        num_vals.append(float(em[ek]))
                    except (ValueError, TypeError):
                        pass
            if num_vals:
                extra_stats[ek] = (float(np.mean(num_vals)), float(np.std(num_vals)))

        param_desc = ", ".join([f"`{k}={v}`" for k, v in params.items()]) or f"Config #{idx}"

        processed_combos.append({
            "combo_index": idx,
            "params": params,
            "param_desc": param_desc,
            "mean_best_val": mean_best_val,
            "std_best_val": std_best_val,
            "mean_final_train": mean_final_train,
            "mean_final_val": mean_final_val,
            "extra_stats": extra_stats,
            "run_dir": run_dir,
            "num_trials": len(metrics_list),
        })

    sorted_combos = sorted(
        processed_combos,
        key=lambda c: c["mean_best_val"] if c["mean_best_val"] is not None else float("inf")
    )

    table_headers = ["Rank", "Configuration", "Mean Best Val Loss"]
    table_aligns = [":---:", ":---", ":---:"]

    for ek in all_extra_keys:
        table_headers.append(f"Mean {ek}")
        table_aligns.append(":---:")

    table_headers.extend(["Mean Final Train Loss", "Mean Final Val Loss", "Trials"])
    table_aligns.extend([":---:", ":---:", ":---:"])

    rows = []
    medals = ["🥇 1", "🥈 2", "🥉 3"]
    for rank_idx, combo in enumerate(sorted_combos, 1):
        rank_badge = medals[rank_idx - 1] if rank_idx <= 3 else str(rank_idx)
        val_str = f"`{combo['mean_best_val']:.4f} ± {combo['std_best_val']:.4f}`" if combo['mean_best_val'] is not None else "`N/A`"
        tr_str = f"`{combo['mean_final_train']:.4f}`" if combo['mean_final_train'] is not None else "`N/A`"
        fv_str = f"`{combo['mean_final_val']:.4f}`" if combo['mean_final_val'] is not None else "`N/A`"

        cells = [rank_badge, combo["param_desc"], val_str]
        for ek in all_extra_keys:
            if ek in combo["extra_stats"]:
                m_v, s_v = combo["extra_stats"][ek]
                cells.append(f"`{m_v:.4f} ± {s_v:.4f}`")
            else:
                cells.append("`N/A`")

        cells.extend([tr_str, fv_str, f"`{combo['num_trials']}`"])
        rows.append("| " + " | ".join(cells) + " |")

    leaderboard_table = f"| {' | '.join(table_headers)} |\n| {' | '.join(table_aligns)} |\n" + "\n".join(rows)

    top_performer = sorted_combos[0] if sorted_combos else None
    if top_performer and top_performer["mean_best_val"] is not None:
        best_details = [f"- **Optimal Configuration:** {top_performer['param_desc']}"]
        best_details.append(f"- **Best Validation Loss:** `{top_performer['mean_best_val']:.4f} ± {top_performer['std_best_val']:.4f}`")
        for ek, (mv, sv) in top_performer["extra_stats"].items():
            best_details.append(f"- **Best {ek}:** `{mv:.4f} ± {sv:.4f}`")
        winner_box = "\n".join(best_details)
    else:
        winner_box = "- No winning configuration determined."

    exp_bullet = f"- **Experiment Name:** `{exp_name}`\n" if exp_name else ""
    start_line = f"- **Start Date:** `{start_time}`\n" if start_time else f"- **Execution Date:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n"
    end_line = f"- **End Date:** `{end_time}`\n" if end_time else ""
    dur_line = f"- **Total Duration:** `{utils.format_duration(duration_seconds)}` (`{duration_seconds:.2f}s`)\n" if duration_seconds is not None else ""
    dev_str = _resolve_device_string(config_dict)
    dev_line = f"- **Execution Device:** `{dev_str}`\n"
    total_configs = len(sweep_results)

    report_md = f"""# {header_title}

## Study Overview
{exp_bullet}{start_line}{end_line}{dur_line}{dev_line}- **Configurations Tested:** `{total_configs}`
- **Total Executions:** `{sum(c['num_trials'] for c in processed_combos)}` trials

## Executive Summary
{winner_box}

## Comparative Leaderboard
{leaderboard_table}

## Base Experiment Configuration
| Parameter | Value |
| :--- | :--- |
{param_rows}
"""
    _write_report(out_file, report_md)
    return report_md



