<a id="readme-top"></a>

# BoKeTE

[![Licence: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch: 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Build System: Hatchling](https://img.shields.io/badge/Build-Hatchling-green.svg)](https://hatch.pypa.io/)

A minimal PyTorch training and experimentation helper library for my personal use. Extracted from my computer vision final year project (2024/2025) at the University of Nottingham. Named after the Bad Bunny song *BoKeTe*, which refers to pothole in English. Designed as a modular, reusable helper package for general PyTorch deep learning workflows:

<p align="center">
  <img src="assets/BoKeTE.png" alt="BoKeTE Music Video" width="600" />
  <br>
  <sub><em>"BoKeTe": Music video homage & namesake inspiration.</em></sub>
</p>

- **Training & Execution**: `Trainer` loop with mixed-precision (AMP), auto cuDNN benchmarking, `EarlyStopping`, and model/optimizer checkpointing.
- **Experimentation & Reporting**: Metric tracking, loss curve rendering (Matplotlib + Chart.js HTML), grid search parameter sweeps, and GFM Markdown report compilation (single-trial + multi-trial summaries).

> [!NOTE]
> **Personal Tooling & Disclaimer**: `bokete` is an opinionated, personal PyTorch helper library created to streamline my own research workflows. While open-sourced under the MIT License, it is provided "as is" without warranty, guarantee of support, or promises of backward compatibility. If you choose to use it, you do so entirely at your own risk.

## Table of Contents

- [1. Public API Summary](#1-public-api-summary)
- [2. Installation](#2-installation)
- [3. Core Modules & API Reference](#3-core-modules--api-reference)
  - [3.1 Training & Checkpointing (`bokete.training`)](#31-training--checkpointing-boketetraining)
  - [3.2 Metric Tracking & Summaries (`bokete.metrics`)](#32-metric-tracking--summaries-boketemetrics)
  - [3.3 Loss Curve Plotting (`bokete.plotting`)](#33-loss-curve-plotting-boketeplotting)
  - [3.4 Markdown Experiment Reporting (`bokete.reporting`)](#34-markdown-experiment-reporting-boketereporting)
  - [3.5 Hyperparameter Sweeps (`bokete.experiments`)](#35-hyperparameter-sweeps-boketeexperiments)
- [4. Usage Example](#4-usage-example)
- [5. Project Architecture](#5-project-architecture)

---

## 1. Public API Summary

All core primitives are exported at the root package level (`from bokete import ...`):

| Function / Class | Module | Description |
| :--- | :--- | :--- |
| `set_seed(seed, deterministic=False)` | `bokete.training` | Seeds Python, NumPy, and PyTorch (CPU/CUDA) for reproducibility. |
| `determine_device()` | `bokete.training` | Detects best available hardware device (`cuda`, `mps`, or `cpu`). |
| `Trainer(...)` | `bokete.training` | Main training loop wrapper with AMP mixed-precision, auto-cuDNN benchmarking & gradient clipping. |
| `EarlyStopping(...)` | `bokete.training` | Callback signaling early stop when validation loss stalls. |
| `Checkpoint(...)` | `bokete.training` | Callback saving `best.pt` and `last.pt` model weights + optimizer state. |
| `training_report(metrics)` | `bokete.metrics` | Computes final losses, mean losses, and best epoch summary dict. |
| `plot_loss_curves(...)` | `bokete.plotting` | Renders Matplotlib loss graph & interactive Chart.js HTML plot. |
| `experiment_report(...)` | `bokete.reporting` | Generates a structured GFM Markdown trial report string with dynamic overview metadata. |
| `multi_trial_report(...)` | `bokete.reporting` | Generates a combined GFM Markdown summary report string across multiple experiment trials. |
| `run_experiments(...)` | `bokete.experiments` | Executes grid search parameter sweeps across configuration paths. |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 2. Installation

`bokete` is packaged using the `hatchling` build backend and is fully compatible with `uv` workspaces.

### Workspace / Local Installation (`uv`)
When used inside a project workspace, `uv sync` automatically installs `bokete` in editable mode:

```bash
uv sync
```

*Or standalone via `uv pip`:*
```bash
uv pip install -e ./bokete
```

### Dependency Requirements
- **Python**: `>= 3.10`
- **PyTorch**: `>= 2.0`
- **NumPy**, **Matplotlib**, **tqdm**

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 3. Core Modules & API Reference

### 3.1 Training & Checkpointing (`bokete.training`)
Provides the core training loop wrapper (`Trainer`), early stopping logic (`EarlyStopping`), model checkpointing (`Checkpoint`), and hardware device detection (`determine_device`).

```python
from bokete import Trainer, EarlyStopping, Checkpoint, determine_device, set_seed

# Reproducibility & Hardware setup
set_seed(42)
device = determine_device()

# Setup callbacks
early_stop = EarlyStopping(patience=10, min_delta=1e-4)
checkpoint = Checkpoint(directory="checkpoints")

# Initialise and execute training loop
trainer = Trainer(model, criterion, optimizer, device=device, amp=True)
metrics = trainer.fit(
    train_loader, 
    val_loader, 
    epochs=50, 
    early_stopping=early_stop,
    checkpoint=checkpoint
)
```

### 3.2 Metric Tracking & Summaries (`bokete.metrics`)
Aggregates running loss metrics per epoch and generates structured summary statistics (final losses, mean losses, best epoch).

```python
from bokete import TrainingMetrics, training_report

# Compute summary stats from training history dictionary
summary = training_report(eval_metrics)
# Returns: {'final_train_loss': 0.12, 'final_val_loss': 0.18, 'mean_train_loss': ..., 'best_epoch': 34}
```

### 3.3 Loss Curve Plotting (`bokete.plotting`)
Renders publication-ready training and validation loss curves using Matplotlib with annotated best-epoch markers, alongside a companion interactive Chart.js HTML file.

```python
from bokete import plot_loss_curves

plot_loss_curves(
    train_loss=eval_metrics['train_loss'],
    val_loss=eval_metrics['val_loss'],
    path="results/graph.png",
    best_epoch=summary['best_epoch']
)
```

### 3.4 Markdown Experiment Reporting (`bokete.reporting`)
Generates structured, self-contained GitHub-Flavoured Markdown (`.md`) reports containing trial overviews, key hyperparameter highlights, loss graph embeds, custom evaluation metrics, and multi-trial statistical summaries ($\text{Mean} \pm \text{Std}$, $\text{Min}$, $\text{Max}$).

```python
from bokete import experiment_report, multi_trial_report

# 1. Single Trial Report
md_report = experiment_report(
    config=config_dict,
    metrics_summary=summary,
    train_loss=eval_metrics['train_loss'],
    val_loss=eval_metrics['val_loss'],
    graph_filename="graph.png",
    title="Trial 1 Report [mixed_r0.1_s0.1]",
    extra_metrics={"Convergence Speed": 0.2145}
)

with open("result.md", "w", encoding="utf-8") as f:
    f.write(md_report)

# 2. Multi-Trial Combined Summary Report
summary_md = multi_trial_report(
    config=config_dict,
    all_trial_metrics=[metrics_trial_1, metrics_trial_2, metrics_trial_3]
)

with open("summary-report.md", "w", encoding="utf-8") as f:
    f.write(summary_md)
```

### 3.5 Hyperparameter Sweeps (`bokete.experiments`)
Orchestrates parameter sweeps over a grid of configuration paths without bleeding state across runs.

```python
from bokete import run_experiments

param_grid = {
    "training.lr": [0.001, 0.0001],
    "training.batch_size": [8, 16]
}

results = run_experiments(
    base_config=config,
    param_grid=param_grid,
    run_fn=train_and_eval_callback
)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 4. Usage Example

Here is a complete, minimal example combining `bokete` helpers inside a standard PyTorch pipeline:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from bokete import (
    determine_device, 
    set_seed, 
    Trainer, 
    training_report, 
    plot_loss_curves, 
    experiment_report
)

# 1. Setup Environment
set_seed(42)
device = determine_device()

# 2. Model, Loss, Optimizer
model = MyNeuralNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 3. Train using Trainer wrapper
trainer = Trainer(model, criterion, optimizer, device=device)
history = trainer.fit(train_loader, val_loader, epochs=20)

# 4. Generate Reports & Visualisations
summary = training_report(history.as_dict())
plot_loss_curves(
    train_loss=history.train_loss,
    val_loss=history.val_loss,
    path="graph.png",
    best_epoch=summary['best_epoch']
)

md = experiment_report(
    config={"lr": 1e-3, "epochs": 20},
    metrics_summary=summary,
    train_loss=history.train_loss,
    val_loss=history.val_loss,
    graph_filename="graph.png",
    title="Minimal Training Run"
)

with open("result.md", "w", encoding="utf-8") as f:
    f.write(md)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 5. Project Architecture

```text
bokete/
├── pyproject.toml         # Hatchling build system configuration
├── README.md              # Package documentation
└── src/
    └── bokete/
        ├── __init__.py    # Public API exports
        ├── experiments.py # Grid sweep orchestration
        ├── metrics.py     # Loss aggregation & summary statistics
        ├── plotting.py    # Matplotlib loss curve & Chart.js HTML rendering
        ├── reporting.py   # Single-trial & multi-trial Markdown report generation
        └── training.py    # Core Trainer, EarlyStopping, Checkpoint, auto-benchmarking
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>
