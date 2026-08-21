<a id="readme-top"></a>

# BOKeTE

[![PyPI Version](https://img.shields.io/pypi/v/BOKeTE.svg)](https://pypi.org/project/BOKeTE/)
[![Licence: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch: 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Build System: Hatchling](https://img.shields.io/badge/Build-Hatchling-green.svg)](https://hatch.pypa.io/)

A minimal PyTorch training and experimentation helper library for my personal use. Extracted from my computer vision final year project (2024/2025) at the University of Nottingham. Named after the Bad Bunny song *BoKeTe*, which refers to pothole in English. Designed as a modular, reusable helper package for general PyTorch deep learning workflows:

<p align="center">
  <img src="https://raw.githubusercontent.com/junhaochai/BoKeTE/master/assets/BOKeTE.png" alt="BOKeTE Music Video" width="600" />
  <br>
  <sub><em>"BOKeTE": Music video homage & namesake inspiration.</em></sub>
</p>

- **Training & Execution**: `Trainer` loop with mixed-precision (AMP), auto cuDNN benchmarking, `EarlyStopping`, and model/optimizer checkpointing.
- **Experimentation & Reporting**: Metric tracking, loss curve rendering (Matplotlib + Chart.js HTML), grid search parameter sweeps, and GFM Markdown report compilation (single-trial + multi-trial summaries).

> [!NOTE]
> **Personal Tooling & Disclaimer**: `bokete` is an opinionated, personal PyTorch helper library created to streamline my own research workflows. While open-sourced under the MIT License, it is provided "as is" without warranty, guarantee of support, or promises of backward compatibility. If you choose to use it, you do so entirely at your own risk.

## Table of Contents

- [1. Public API Summary](#1-public-api-summary)
- [2. Installation](#2-installation)
- [3. Complete Usage Example](#3-complete-usage-example)
- [4. Advanced Workflows & Feature Reference](#4-advanced-workflows--feature-reference)
  - [4.1 Loss Curve Plotting (`bokete.plotting`)](#41-loss-curve-plotting-boketeplotting)
  - [4.2 Multi-Trial Execution & Sweeps (`bokete.experiments`)](#42-multi-trial-execution--sweeps-boketeexperiments)
  - [4.3 Configuration & Directory Utilities (`bokete.utils`)](#43-configuration--directory-utilities-boketeutils)
- [5. Project Architecture](#5-project-architecture)

---

## 1. Public API Summary

All core primitives are exported at the root package level (`from bokete import ...`):

| Function / Class | Module | Description |
| :--- | :--- | :--- |
| `load_config(config_path)` | `bokete.utils` | Loads a `.yaml`, `.yml`, or `.json` file into a dictionary with standardized logging. |
| `set_seed(seed, deterministic=False)` | `bokete.training` | Seeds Python, NumPy, and PyTorch (CPU/CUDA) for reproducibility. |
| `determine_device()` | `bokete.training` | Detects best available hardware device (`cuda`, `mps`, or `cpu`). |
| `Trainer(...)` | `bokete.training` | Main training loop wrapper with AMP mixed-precision, auto-cuDNN benchmarking & gradient clipping. |
| `EarlyStopping(...)` | `bokete.training` | Callback signaling early stop when validation loss stalls. |
| `Checkpoint(...)` | `bokete.training` | Callback saving `best.pt` and `last.pt` model weights + optimizer state. |
| `training_report(metrics)` | `bokete.metrics` | Computes final losses, mean losses, and best epoch summary dict. |
| `plot_loss_curves(...)` | `bokete.plotting` | Renders Matplotlib loss graph & interactive Chart.js HTML plot. |
| `experiment_report(...)` | `bokete.reporting` | Generates a structured GFM Markdown trial report string with dynamic overview metadata. |
| `multi_trial_report(...)` | `bokete.reporting` | Generates a combined GFM Markdown summary report string across multiple experiment trials. |
| `log_trial_start(...)` | `bokete.utils` | Logs a standardized trial header with clean unformatted console spacing. |
| `run_trials(...)` | `bokete.experiments` | Orchestrates multi-trial runs with clean logging, Ctrl+C cancellation, and auto-report saving. |
| `run_experiments(...)` | `bokete.experiments` | Executes grid search parameter sweeps across configuration paths. |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 2. Installation

Install `bokete` directly from PyPI:

```bash
pip install bokete
```

*Or via `uv`:*
```bash
uv pip install bokete
```

### Local / Editable Installation
For local development and workspace integration:

```bash
uv pip install -e .
```

### Dependency Requirements
- **Python**: `>= 3.10`
- **PyTorch**: `>= 2.0`
- **NumPy**, **Matplotlib**, **tqdm**

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 3. Complete Usage Example

`bokete` is designed to pair seamlessly with structured YAML configuration files. Below is a complete, real-world example combining a YAML config template and a PyTorch training pipeline:

### 3.1 Recommended Configuration Template (`configs/config.yaml`)

```yaml
# ==============================================================================
# Experiment Configuration (configs/config.yaml)
# ==============================================================================

# Global Execution Settings
experiment_name: "baseline_classification_v1"
seed: 42
device: null           # Hardware target: "cuda", "cpu", or null (auto-detect)

# Dataset Configuration
dataset:
  name: "CIFAR10"
  prop_train: 0.8

# Architecture & Layer Configuration
model:
  arch: "mlp"
  num_layers: 4
  hidden_features: 128
  out_features: 10

# Training Hyperparameters & Features
training:
  epochs: 20
  batch_size: 32
  optimizer: "Adam"
  lr: 0.001
  amp: false                     # Mixed precision (FP16 on CUDA)
  max_grad_norm: null            # L2 gradient clipping threshold
  early_stopping_patience: 10   # Stop if val loss stalls
  save_checkpoints: true         # Save best.pt and last.pt weights
```

### 3.2 PyTorch Pipeline Integration (`train.py`)

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import bokete

# 1. Load YAML Config & Setup Environment
config = bokete.load_config("configs/config.yaml")
t_cfg = config.get("training", {})

bokete.set_seed(config.get("seed", 42))
device = config.get("device") or bokete.determine_device()

# 2. Setup Model, Loss, Optimizer & Callbacks
model = MyNeuralNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg.get("lr", 1e-3))

callbacks = []
if t_cfg.get("early_stopping_patience"):
    callbacks.append(bokete.EarlyStopping(patience=t_cfg["early_stopping_patience"]))
if t_cfg.get("save_checkpoints"):
    callbacks.append(bokete.Checkpoint(directory="checkpoints"))

# 3. Train Model via Trainer Wrapper
trainer = bokete.Trainer(
    model=model, 
    criterion=criterion, 
    optimizer=optimizer, 
    device=device, 
    amp=t_cfg.get("amp", False),
    max_grad_norm=t_cfg.get("max_grad_norm")
)

history = trainer.fit(
    train_loader=train_loader, 
    val_loader=val_loader, 
    epochs=t_cfg.get("epochs", 20), 
    callbacks=callbacks
)

# 4. Generate & Save Markdown Report (Auto-embeds YAML table & loss curves!)
bokete.experiment_report(
    config=config,
    metrics=history,
    save_path="results/report.md",
    title="Minimal Training Run"
)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 4. Advanced Workflows & Feature Reference

### 4.1 Loss Curve Plotting (`bokete.plotting`)
Renders publication-ready training and validation loss curves using Matplotlib with annotated best-epoch markers, alongside a companion interactive Chart.js HTML file. Accepts a `TrainingMetrics` instance directly or raw loss lists.

```python
import bokete

# Pass the TrainingMetrics object directly!
bokete.plot_loss_curves(metrics=history, path="results/graph.png")
```

### 4.2 Multi-Trial Execution & Sweeps (`bokete.experiments`)
    save_path="results/report.md"
)
```

### Option B: Recommended High-Level Runner (`bokete.run_experiments`)

```python
import bokete

# 1. Define factory functions
def model_factory(cfg):
    return torch.nn.Linear(10, 1)

def loader_factory(cfg):
    x, y = torch.randn(100, 10), torch.randn(100, 1)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=cfg["batch_size"])
    return loader, loader

# 2. Build automated trial runner
runner = bokete.create_trial_runner(model_factory, loader_factory)

# 3. Load config and run experiment (auto-detects single config vs grid sweep)
cfg = bokete.parse_cli_config(MyConfigSchema)
bokete.run_experiments(config=cfg, run_fn=runner)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 4. Feature Reference & Advanced API

### 4.1 Orchestration & Sweeps (`bokete.run_experiments`)

`bokete.run_experiments` is the primary top-level entry point. It automatically detects whether your config contains a `param_grid`:

#### Single-Config Multi-Trial Run
Runs a single configuration across multiple seed trials with clean console logging, graceful `Ctrl+C` cancellation, and auto-generation of `summary-report.md`:

```python
import bokete

all_trial_metrics = bokete.run_experiments(
    config=config,
    num_trials=3,
    run_fn=runner
)
```

#### Grid Search Sweeps
Executes parameter combinations across a configuration grid:

```python
import bokete

config = {
    "experiment_name": "grid_sweep_demo",
    "trials": 2,
    "param_grid": {
        "training.lr": [0.001, 0.0001],
        "model.hidden_dim": [64, 128],
    }
}

results = bokete.run_experiments(config=config, run_fn=runner)
```

### 4.2 Loss Curve Plotting (`bokete.plotting`)
Renders publication-ready training and validation loss curves using Matplotlib with annotated best-epoch markers, alongside a companion interactive Chart.js HTML file. Accepts a `TrainingMetrics` instance directly or raw loss lists.

```python
import bokete

# Pass the TrainingMetrics object directly!
bokete.plot_loss_curves(metrics=history, path="results/graph.png")
```

### 4.3 Configuration & Directory Utilities (`bokete.utils`)
Provides configuration loading (`load_config`), experiment run directory creation (`create_run_directory`), and nested key dictionary utilities (`flatten_dict`, `set_nested_key`).

```python
import bokete

# 1. Load YAML or JSON configuration file into a dictionary
config = bokete.load_config("configs/config.yaml")

# 2. Programmatically override nested configuration keys using dot notation
bokete.set_nested_key(config, "training.lr", 0.0005)

# 3. Create a timestamped experiment output directory (e.g. results/exp_name/YYYYMMDD-01)
run_dir = bokete.create_run_directory(base_dir="results", experiment_name=config.get("experiment_name"))
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
        ├── experiments.py # Multi-trial execution & grid search parameter sweeps
        ├── metrics.py     # Loss aggregation & summary statistics
        ├── plotting.py    # Matplotlib loss curve & Chart.js HTML rendering
        ├── reporting.py   # Single-trial & multi-trial Markdown report generation
        ├── training.py    # Core Trainer, EarlyStopping, Checkpoint, auto-benchmarking
        └── utils.py       # Configuration I/O, device checks, seeds & console formatters
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>
