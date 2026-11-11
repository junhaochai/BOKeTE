<a id="readme-top"></a>

# BOKeTE

[![PyPI Version](https://img.shields.io/pypi/v/BOKeTE.svg)](https://pypi.org/project/BOKeTE/)
[![Licence: All rights reserved](https://img.shields.io/badge/Licence-All_rights_reserved-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch: 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Build System: Hatchling](https://img.shields.io/badge/Build-Hatchling-green.svg)](https://hatch.pypa.io/)

A minimal PyTorch training and experimentation helper library for my personal use. Extracted from my computer vision final year project (2024/2025) at the University of Nottingham. Named after the Bad Bunny song *BoKeTe*, which refers to pothole in English. Designed as a modular, reusable helper package for general PyTorch deep learning workflows:

<p align="center">
  <img src="https://raw.githubusercontent.com/junhaochai/BoKeTE/master/assets/BOKeTE.png" alt="BOKeTE Music Video" width="600" />
  <br>
  <sub><em>"BOKeTE": Music video homage & namesake inspiration.</em></sub>
</p>

- **Training & Execution**: `Trainer` loop with mixed-precision (AMP), auto cuDNN benchmarking, `EarlyStopping`, and full-state checkpoint resume.
- **Experimentation & Reporting**: Metric tracking, Matplotlib loss curves, grid search parameter sweeps, and GFM Markdown report compilation (single-trial + multi-trial summaries).

> [!NOTE]
> **Personal Tooling & Permissions**: `bokete` is a personal PyTorch helper library made publicly visible for portfolio review and inspection. From `0.2.0` onwards, including `0.2.0.dev0`, reuse requires prior written permission from [Jun Hao Chai](https://github.com/junhaochai), except where applicable law, hosting-platform terms or existing licence grants permit otherwise. Earlier MIT-licensed versions and material retain their original permissions; their notices remain in the earlier releases and Git history. See [LICENSE](LICENSE) for current terms. The software is provided "as is", without warranty, guaranteed support or promises of backward compatibility.

## Table of Contents

- [1. Public API Summary](#1-public-api-summary)
- [2. Installation](#2-installation)
- [3. Complete Usage Example](#3-complete-usage-example)
- [4. Advanced Workflows & Feature Reference](#4-advanced-workflows--feature-reference)
  - [4.1 Loss Curve Plotting (`bokete.plotting`)](#41-loss-curve-plotting-boketeplotting)
  - [4.2 Multi-Trial Execution & Sweeps (`bokete.experiments`)](#42-multi-trial-execution--sweeps-boketeexperiments)
  - [4.3 Configuration & Directory Utilities (`bokete.utils`)](#43-configuration--directory-utilities-boketeutils)
  - [4.4 Validation Metric Selection](#44-validation-metric-selection)
  - [4.5 Checkpoint Resume](#45-checkpoint-resume)
  - [4.6 Interrupted Experiments](#46-interrupted-experiments)
  - [4.7 Reproducibility & Runtime Records](#47-reproducibility--runtime-records)
- [5. Project Architecture](#5-project-architecture)

---

## 1. Public API Summary

All core primitives are exported at the root package level (`from bokete import ...`):

| Function / Class | Module | Description |
| :--- | :--- | :--- |
| `load_config(config_path)` | `bokete.utils` | Loads a `.yaml` or `.yml` file into a dictionary with standardised logging. |
| `resolve_config_path(path)` | `bokete.utils` | Resolves config paths inside `configs/` directory (auto-appending `.yaml` for short names). |
| `set_seed(seed, deterministic=False)` | `bokete.utils` | Seeds Python, NumPy, and PyTorch (CPU/CUDA) for reproducibility. |
| `determine_device()` | `bokete.utils` | Detects best available hardware device (`cuda`, `mps`, or `cpu`). |
| `Trainer(...)` | `bokete.training` | Main training loop wrapper with AMP mixed-precision, auto-cuDNN benchmarking & gradient clipping. |
| `EarlyStopping(...)` | `bokete.training` | Callback signaling early stop when validation loss stalls. |
| `Checkpoint(...)` | `bokete.training` | Callback saving `best.pt` and `last.pt`, including full training state when used with `Trainer.fit`. |
| `training_report(metrics)` | `bokete.metrics` | Computes final losses, mean losses, and best epoch summary dict. |
| `plot_loss_curves(...)` | `bokete.plotting` | Renders a Matplotlib loss graph as a PNG. |
| `trial_report(...)` | `bokete.reporting` | Generates a GFM Markdown report for a single training run. |
| `config_report(...)` | `bokete.reporting` | Summarises multiple trials for one configuration. |
| `experiment_report(...)` | `bokete.reporting` | Compares configurations from a parameter sweep. |
| `log_trial_start(...)` | `bokete.utils` | Logs a standardized trial header with clean unformatted console spacing. |
| `run_experiments(...)` | `bokete.experiments` | Primary top-level entry point for single-trial runs, multi-trial runs, and grid search sweeps. |

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
  monitor: "val_accuracy"        # Higher wins; val_loss selects the lowest loss
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
    callbacks=callbacks,
    monitor=t_cfg.get("monitor", "val_loss"),
)

# 4. Generate & Save Markdown Report (Auto-embeds YAML table & loss curves!)
bokete.trial_report(
    config=config,
    metrics=history,
    save_path="results/report.md",
    title="Minimal Training Run"
)
```

### 3.3 Running the Included Starter Template (`example/`)

The repository includes a ready-to-run starter template under [`example/`](file:///D:/la-mudanza/BoKeTE/example/main.py) demonstrating `bokete.parse_cli_config`, `bokete.create_trial_runner`, and `bokete.run_experiments`.

Execute the starter template directly with `uv`:

```bash
uv --directory example run python main.py --config example
```

*(Or from inside `example/`: `uv run python main.py --config example`)*

To trigger an automated hyperparameter grid sweep, uncomment `param_grid` in [`example/configs/example.yaml`](file:///D:/la-mudanza/BoKeTE/example/configs/example.yaml).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 4. Advanced Workflows & Feature Reference

### 4.1 Loss Curve Plotting (`bokete.plotting`)

Renders training and validation loss curves as PNG files using Matplotlib with annotated best-epoch markers. Accepts a `TrainingMetrics` instance directly or raw loss lists.

```python
import bokete

# Pass the TrainingMetrics object directly!
bokete.plot_loss_curves(metrics=history, path="results/graph.png")
```

### 4.2 Multi-Trial Execution & Sweeps (`bokete.experiments`)

`bokete.run_experiments` is the primary top-level entry point. It automatically detects whether your config contains a `param_grid`.

#### Building a Trial Runner

```python
import bokete
import torch
from torch.utils.data import DataLoader, TensorDataset

# 1. Define factory functions
def model_factory(cfg):
    return torch.nn.Linear(10, 2)

def loader_factory(cfg):
    x, y = torch.randn(100, 10), torch.randint(0, 2, (100,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=cfg.get("training", {}).get("batch_size", 32))
    return loader, loader

# 2. Build automated trial runner
runner = bokete.create_trial_runner(model_factory, loader_factory)

# 3. Load config and run experiment (auto-detects single config vs grid sweep)
config = bokete.load_config("configs/config.yaml")
bokete.run_experiments(config=config, run_fn=runner)
```

#### Single-Config Multi-Trial Run

Runs a single configuration across multiple seed trials with clean console logging, graceful `Ctrl+C` cancellation, and auto-generation of `config-report.md`:

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

### 4.3 Configuration & Directory Utilities (`bokete.utils`)

Provides configuration loading (`load_config`), experiment run directory creation (`create_run_directory`), and nested key dictionary utilities (`flatten_dict`, `set_nested_key`).

```python
import bokete

# 1. Load a YAML configuration file into a dictionary
config = bokete.load_config("configs/config.yaml")

# 2. Programmatically override nested configuration keys using dot notation
bokete.set_nested_key(config, "training.lr", 0.0005)

# 3. Create a timestamped experiment output directory (e.g. results/exp_name/YYYYMMDD-01)
run_dir = bokete.create_run_directory(base_dir="results", experiment_name=config.get("experiment_name"))
```

### 4.4 Validation Metric Selection

Set `training.monitor` when using `create_trial_runner`, or pass `monitor`
directly to `Trainer.fit`:

```yaml
training:
  monitor: val_accuracy
```

- `val_accuracy` selects the highest validation top-1 accuracy. It requires
  multiclass outputs shaped `[batch, classes]` and integer class-index targets
  shaped `[batch]`. Accuracy is a fraction from 0 to 1, calculated across all
  samples during the existing validation pass, and is collected when selected.
  Targets matching the loss function's `ignore_index` are excluded. Batches
  with no valid targets are skipped; validation requires at least one valid target.
- `val_loss` selects the lowest validation loss and remains the default for
  configurations that omit `monitor`. It also supports regression models.
- The direction is inferred; no `mode` setting is needed. Ties retain the
  earlier epoch. Checkpoints, early stopping and sweep rankings use the selected
  metric. Early stopping's `min_delta` controls how much improvement resets
  patience; checkpoints save every strict improvement.
- Reports identify the selected epoch and its validation score. Minimum loss
  can occur at a different epoch when accuracy is monitored. Test callbacks
  remain separate and do not select checkpoints.
- A sweep must use the same monitor across configurations so its scores can be
  compared. A `ReduceLROnPlateau` scheduler must use `mode="max"` for accuracy
  or `mode="min"` for loss; it receives the monitored score.
- Epoch losses pool batch totals: `mean` uses element counts for elementwise
  losses and valid target counts or class-weight sums for cross-entropy/NLL;
  `batchmean` uses sample counts, and `sum` remains a sum. Custom scalar criteria
  are treated as per-sample means, using Tensor targets or outputs to infer
  batch size; fully opaque structures retain one unit of weight per batch.
  Ignored-only batches are skipped, and
  validation with no valid observations is rejected for either monitor.
  A mean-reduced training batch with zero total class weight is rejected
  before the optimiser step because its mean loss is undefined.

### 4.5 Checkpoint Resume

Pass a checkpoint path to `Trainer.fit(resume_from=...)` to continue from a
completed epoch. The default, `resume_from=None`, starts a fresh run.

```python
history = trainer.fit(
    train_loader,
    val_loader,
    epochs=20,
    scheduler=scheduler,
    early_stopping=early_stopping,
    checkpoint=bokete.Checkpoint("checkpoints"),
    resume_from="checkpoints/last.pt",
)
```

- `epochs` is the total target, including saved epochs. Resuming after 10
  completed epochs with `epochs=20` runs epochs 11 through 20. A checkpoint at
  or beyond the target performs no further training. A checkpoint whose early
  stopping already fired remains stopped.
- `last.pt` continues the latest saved epoch; `best.pt` continues the selected
  best epoch, which may be earlier. Both contain model and optimiser state,
  scheduler and AMP scaler state, best score, early-stopping policy and counter,
  completed epoch, metric history and RNG state. Checkpoints are written after
  the other epoch-end callbacks finish.
- Construct the matching model, optimiser, scheduler and callbacks before
  calling `fit`. Use the same monitored metric, AMP setting, stateful callback
  types/order, loader generator configuration and number of visible CUDA
  devices. Callbacks supplied through `callbacks` are also restored when they
  provide `state_dict()` and `load_state_dict()`; their saved state must contain
  tensors and plain Python values compatible with `weights_only=True` loading.
- RNG restoration covers Python, NumPy's global generator, PyTorch CPU/CUDA
  generators, and explicit training/validation DataLoader and sampler
  generators. Independently owned RNG objects, custom sampler/dataset state,
  persistent worker RNG state and distributed per-rank state are not captured.
  Identical numerical results also require a matching runtime and deterministic
  operations. Resume starts between epochs, rather than partway through a batch.
- Returned histories include the saved epochs, so reports and plots retain
  their original epoch numbering. Timing describes the current `fit` call.
- Older checkpoints and standalone `Checkpoint.update()` saves remain usable
  for loading weights, but lack the full state required by `resume_from`.
- Each checkpoint file is written to a sibling temporary file and replaced
  only after serialisation succeeds. `best.pt` and `last.pt` are independent
  writes, so a save exception can leave them at different epochs; recovery
  requires checking both files and choosing the intended checkpoint explicitly.

#### Resume Through the Experiment Runner

Map each trial number to its own full-state checkpoint in the configuration:

```yaml
experiment_name: baseline_classification_v1
trials: 3
deterministic: false
training:
  epochs: 20
  resume_from:
    "1": "results/previous/trial_1/checkpoints/last.pt"
    "2": "results/previous/trial_2/checkpoints/last.pt"
  # Optional when best.pt was moved or renamed:
  resume_best_from:
    "2": "results/archive/trial_2-best.pt"
```

- Trial 2 resumes its own state; it does not continue from trial 1. Unlisted
  trials start fresh. A completed trial can be listed with its checkpoint at
  the target epoch to retain its history without training it again. Use the
  same seed, dataset/model configuration and deterministic policy as before.
- Paths are relative to the process working directory, or may be absolute.
  Each resumed invocation normally creates a new output directory. It does
  not discover old runs or skip completed trials automatically.
- For a parameter sweep, use quoted `"configuration:trial"` keys instead,
  for example `"2:1"` means the second parameter combination's first trial.
  Both numbers start at 1 and follow the configured grid order. Keep that
  order unchanged when resuming. Trial-only keys are rejected in sweeps to
  prevent sharing one trial's checkpoint across different configurations.
- The runner retains a matching original best checkpoint in the new trial's
  checkpoint directory when saving is enabled. If `resume_best_from` is
  omitted, it checks the resume checkpoint itself when it is the selected
  epoch, otherwise its neighbouring `best.pt`. The candidate must match the
  selected epoch, score and monitored metric in the saved history. Keep
  checkpoint paths within the intended trial; matching scores alone do not
  establish a file's provenance.
- If no matching original best checkpoint is available and training makes
  no new improvement, reports mark it unavailable and omit best-weight test
  metrics. Final-weight metrics can still be evaluated. Explicit missing or
  mismatched `resume_best_from` paths raise an error.

Pass `scheduler_factory` to create one scheduler per trial, after constructing
the optimiser and before restoring its state:

```python
def scheduler_factory(optimizer, config):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

runner = bokete.create_trial_runner(
    model_factory,
    loader_factory,
    scheduler_factory=scheduler_factory,
)
```

`step_size=5` reduces the learning rate every five epochs; `gamma=0.5`
halves it at each reduction. Omitting the factory uses no scheduler. A
`ReduceLROnPlateau` factory is an alternative for score-based scheduling;
its mode must match the monitored metric (`min` for loss, `max` for accuracy).
Supply a matching factory when resuming a run that used a scheduler.

### 4.6 Interrupted Experiments

- `run_experiments` stops on `Ctrl+C` and returns completed trials plus an
  interrupted record for the current trial. Records include `trial`,
  `status` (`completed` or `interrupted`), `interrupted` and `run_dir`.
  Sweeps retain the current configuration and stop later configurations.
- Partial histories contain complete training/validation epoch pairs. An
  interruption before the first completed epoch produces an empty history
  with interrupted status. Existing files and the last saved checkpoint are
  retained; resuming replays work after that saved epoch boundary.
- Reports show interrupted trials, but exclude them from aggregate scores,
  mean loss curves and sweep rankings. No post-training test evaluation is
  started after a training interruption.
- Direct `Trainer.fit` and trial-runner calls still raise `KeyboardInterrupt`.
  Catch `bokete.TrainingInterrupted` and read its `.history` to inspect the
  retained `TrainingMetrics`. Custom runners that raise an ordinary
  `KeyboardInterrupt` produce an interrupted record without partial history.

### 4.7 Reproducibility & Runtime Records

Set top-level `deterministic: true` in experiment configuration, or call
`bokete.set_seed(42, deterministic=True)` before constructing CUDA objects.
This enables PyTorch deterministic algorithms and deterministic cuDNN
execution, disables cuDNN benchmarking, and supplies
`CUBLAS_WORKSPACE_CONFIG=:4096:8` when no workspace setting already exists.
The workspace value selects a cuBLAS workspace layout for repeatable CUDA
matrix operations; an existing caller-provided value is preserved.
Operations without a deterministic implementation raise an error rather
than silently weakening the policy. See [PyTorch's deterministic algorithm
documentation](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html).

`deterministic: false` (the default), or `set_seed(..., deterministic=False)`,
explicitly disables deterministic algorithm restrictions and deterministic
cuDNN execution again. Automatic cuDNN benchmarking is then allowed for
consistent input shapes. Disabling leaves any cuBLAS workspace environment
setting in place; it no longer enforces deterministic algorithms.
These settings affect the whole process. Full-state resume requires the same
recorded deterministic settings; older checkpoints without them cannot
validate that policy. Matching hardware/software, data loading and operations
are still necessary for identical results.

Training histories record Python, BOKeTE, PyTorch, NumPy, CUDA and cuDNN
versions plus deterministic settings and the final cuDNN benchmark setting.
Reports display these recorded values, including distinct recorded
environments across trials. Older histories display `Not recorded` instead
of attributing the report writer's runtime to the original training run.
For resumed training, the environment describes the current invocation.
CUDA AMP checkpoint resume was verified on an NVIDIA GeForce RTX 3060 Laptop
GPU using the example environment's PyTorch `2.6.0+cu124`. The regression test
still skips when run with the root environment's CPU-only PyTorch build.

To run the full suite in the CUDA-enabled example environment from the
repository root, set the cuBLAS workspace before Python starts. Earlier
tests can initialise CUDA before later tests enable deterministic execution:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
.\example\.venv\Scripts\python.exe -m unittest discover -s tests -q
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
        ├── plotting.py    # Matplotlib loss curves saved as PNG files
        ├── reporting.py   # Single-trial & multi-trial Markdown report generation
        ├── training.py    # Core Trainer, EarlyStopping, Checkpoint, auto-benchmarking
        └── utils.py       # Configuration I/O, device checks, seeds & console formatters
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>
