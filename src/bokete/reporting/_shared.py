"""Internal formatting, aggregation and file helpers for Markdown reports."""

from dataclasses import asdict, is_dataclass
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch

import bokete.utils as utils


logger = logging.getLogger("bokete.reporting")


def _extra_metric_keys(histories):
    """Collect extra-metric columns in first-seen order, including text values."""
    keys = []
    for history in histories:
        if isinstance(history, dict) and isinstance(history.get("extra_metrics"), dict):
            for key in history["extra_metrics"]:
                if key not in keys:
                    keys.append(key)

    return keys


def _extra_metric_stats(histories, keys):
    """Calculate mean/std for numeric values in the requested metric columns."""
    stats = {}
    for key in keys:
        values = []
        for history in histories:
            extra = history.get("extra_metrics", {}) if isinstance(history, dict) else {}
            if key in extra:
                try:
                    values.append(float(extra[key]))
                except (ValueError, TypeError):
                    pass

        if values:
            stats[key] = (float(np.mean(values)), float(np.std(values)))

    return stats


def _write_report(out_file: Optional[Path], report_md: str) -> None:
    """Writes report string to out_file and logs completion."""
    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(report_md, encoding="utf-8")
        logger.info(f"[BOKeTE] Saved Report to: {out_file}")


def _runtime_section(histories):
    """Display recorded environments; never substitute the report writer's runtime."""
    environments = []
    for history in histories:
        runtime = history.get("runtime")
        if runtime and runtime not in environments:
            environments.append(runtime)

    if not environments:
        return "\n## Runtime Environment\nNot recorded.\n"

    sections = []
    for index, runtime in enumerate(environments, 1):
        rows = "\n".join(f"| `{key}` | `{value}` |" for key, value in utils.flatten_dict(runtime).items())
        label = f"\n### Environment {index}\n" if len(environments) > 1 else ""
        sections.append(f"{label}\n| Setting | Recorded Value |\n| :--- | :--- |\n{rows}\n")

    return "\n## Runtime Environment\n" + "".join(sections)


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
        info = utils._model_info(model)
        return info["model_name"], info["total_params"], info["trainable_params"], info["structure"]

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
