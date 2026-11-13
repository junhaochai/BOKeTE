# Getting started

`bokete` works with YAML configuration files and existing PyTorch training pipelines. The integration sketch below assumes you provide `MyNeuralNetwork`, `train_loader` and `val_loader`. For a runnable project, use the [included starter template](#running-the-included-starter-template).

## Configuration Template (`configs/config.yaml`)

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

## PyTorch Pipeline Integration (`train.py`)

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

## Running the Included Starter Template

The packaged starter lives in `src/bokete/templates/starter/` and demonstrates `bokete.parse_cli_config`, `bokete.create_trial_runner`, and `bokete.run_experiments`. Generate a separate project so its environment, datasets and results stay outside the template:

```bash
bokete init my-experiment
cd my-experiment
uv run python main.py --config example
```

For local development, run `uv run bokete init my-experiment` from the repository root, then use `uv run --with-editable .. python main.py --config example` from the generated directory. The editable override selects your checkout instead of a published BOKeTE release. Keep the template directory free of generated files.

The supplied configuration already enables a hyperparameter grid sweep with three trials per combination. Remove `param_grid` to run one configuration, and set `trials: 1` for a single trial. Adjust these settings in the generated project's `configs/example.yaml`.
