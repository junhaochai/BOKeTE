"""Markdown leaderboards across experiment configurations."""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import numpy as np

import bokete.metrics as metrics_mod
import bokete.utils as utils

from ._shared import (
    _extra_metric_keys,
    _extra_metric_stats,
    _format_config_table,
    _resolve_device_string,
    _runtime_section,
    _write_report,
)


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
    base_monitor = utils.get_nested_key(config_dict, "training.monitor")
    run_monitors = []
    for result in sweep_results:
        effective_monitor = result.get('parameters', {}).get('training.monitor', base_monitor)
        for history in result.get('metrics', []):
            selection = metrics_mod.selection_summary(history, effective_monitor)
            run_monitors.append(selection['monitor'])

    if len(set(run_monitors)) > 1:
        raise ValueError("Cannot compare runs with different monitoring settings.")

    monitor = run_monitors[0] if run_monitors else base_monitor or 'val_loss'
    mode = metrics_mod.monitor_mode(monitor)
    metric_label = utils.format_metric_name(monitor)
    exp_name = config_dict.get('experiment_name') or config_dict.get('exp_name') or config_dict.get('name')
    header_title = f"Experiment Report: {exp_name}" if exp_name else "Experiment Report"

    # Extract aggregated statistics per combo
    processed_combos = []
    all_extra_keys = _extra_metric_keys(
        history for result in sweep_results for history in result.get("metrics", [])
    )

    for idx, res in enumerate(sweep_results, 1):
        params = res.get("parameters", {})
        all_metrics = res.get("metrics", [])
        metrics_list = [m for m in all_metrics if not m.get("interrupted")]
        run_dir = res.get("run_dir")

        selections = [metrics_mod.selection_summary(m, monitor) for m in metrics_list]
        best_val_losses = [s['best_score'] for s in selections if s['best_score'] is not None]
        mean_best_val = float(np.mean(best_val_losses)) if best_val_losses else None
        std_best_val = float(np.std(best_val_losses)) if best_val_losses else 0.0

        final_train_losses = [m['train_loss'][-1] for m in metrics_list if m and 'train_loss' in m and m['train_loss']]
        mean_final_train = float(np.mean(final_train_losses)) if final_train_losses else None

        final_val_losses = [m['val_loss'][-1] for m in metrics_list if m and 'val_loss' in m and m['val_loss']]
        mean_final_val = float(np.mean(final_val_losses)) if final_val_losses else None

        extra_stats = _extra_metric_stats(metrics_list, all_extra_keys)

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
            "interrupted_trials": sum(bool(m.get('interrupted')) for m in all_metrics),
            "status": res.get("status", "interrupted" if any(m.get('interrupted') for m in all_metrics) else "completed"),
        })

    sorted_combos = sorted(
        processed_combos,
        key=lambda c: (c["mean_best_val"] if mode == 'min' else -c["mean_best_val"])
        if c["mean_best_val"] is not None else float("inf")
    )

    table_headers = ["Rank", "Configuration", f"Mean Best {metric_label}"]
    table_aligns = [":---:", ":---", ":---:"]

    for ek in all_extra_keys:
        fmt_ek = utils.format_metric_name(ek)
        header_name = fmt_ek if fmt_ek.lower().startswith(("mean ", "average ")) else f"Mean {fmt_ek}"
        table_headers.append(header_name)
        table_aligns.append(":---:")

    table_headers.extend(["Mean Final Train Loss", "Mean Final Val Loss", "Completed Trials", "Interrupted Trials", "Status"])
    table_aligns.extend([":---:", ":---:", ":---:", ":---:", ":---:"])

    rows = []
    for rank_idx, combo in enumerate(sorted_combos, 1):
        rank_badge = str(rank_idx) if combo['mean_best_val'] is not None else 'N/A'
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

        cells.extend([tr_str, fv_str, f"`{combo['num_trials']}`", f"`{combo['interrupted_trials']}`", f"`{combo['status']}`"])
        rows.append("| " + " | ".join(cells) + " |")

    leaderboard_table = f"| {' | '.join(table_headers)} |\n| {' | '.join(table_aligns)} |\n" + "\n".join(rows)

    top_performer = sorted_combos[0] if sorted_combos else None
    if top_performer and top_performer["mean_best_val"] is not None:
        best_details = [f"- **Configuration:** {top_performer['param_desc']}"]
        best_details.append(f"- **Best {metric_label}:** `{top_performer['mean_best_val']:.4f} ± {top_performer['std_best_val']:.4f}`")
        for ek, (mv, sv) in top_performer["extra_stats"].items():
            fmt_ek = utils.format_metric_name(ek)
            fmt_ek_lower = fmt_ek.lower()
            # Avoid duplicate prefixes if create_trial_runner or the user already tagged
            # the metric with checkpoint/aggregation context (e.g., 'Best Accuracy', 'Final Test Loss')
            if fmt_ek_lower.startswith(("best ", "final ", "mean ", "lowest ", "highest ", "min ", "max ")):
                label = fmt_ek
            else:
                label = f"Best {fmt_ek}"
            best_details.append(f"- **{label}:** `{mv:.4f} ± {sv:.4f}`")
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
    histories = [m for result in sweep_results for m in result.get('metrics', [])]
    runtime_section = _runtime_section(histories)
    status = "interrupted" if any(c['status'] == 'interrupted' for c in processed_combos) else "completed"

    report_md = f"""# {header_title}

## Study Overview
{exp_bullet}{start_line}{end_line}{dur_line}{dev_line}- **Configurations Tested:** `{total_configs}`
- **Total Executions:** `{sum(c['num_trials'] for c in processed_combos)}` trials
- **Status:** `{status}`
- **Interrupted Trials:** `{sum(c['interrupted_trials'] for c in processed_combos)}` (excluded from rankings)
- **Selection Metric:** `{monitor}` ({'minimise' if mode == 'min' else 'maximise'})

## Optimal Configuration
{winner_box}

## Comparative Leaderboard
{leaderboard_table}
{runtime_section}

## Base Experiment Configuration
| Parameter | Value |
| :--- | :--- |
{param_rows}
"""

    _write_report(out_file, report_md)
    return report_md
