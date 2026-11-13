# User guide

## Loss Curve Plotting (`bokete.plotting`)

Renders training and validation loss curves as PNG files using Matplotlib with annotated best-epoch markers. Accepts a `TrainingMetrics` instance directly or raw loss lists.

```python
import bokete

# Pass the TrainingMetrics object directly!
bokete.plot_loss_curves(metrics=history, path="results/graph.png")
```

## Multi-Trial Execution & Sweeps (`bokete.experiments`)

`bokete.run_experiments` is the primary top-level entry point. It automatically detects whether your config contains a `param_grid`.

### Building a Trial Runner

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

### Single-Config Multi-Trial Run

Runs a single configuration across multiple seed trials with clean console logging, graceful `Ctrl+C` cancellation, and auto-generation of `config-report.md`:

```python
import bokete

all_trial_metrics = bokete.run_experiments(
    config=config,
    num_trials=3,
    run_fn=runner
)
```

### Grid Search Sweeps

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

## Configuration & Directory Utilities (`bokete.utils`)

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

## Validation Metric Selection

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

## Checkpoint Resume

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

### Resume Through the Experiment Runner

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

## Interrupted Experiments

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

## Reproducibility & Runtime Records

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

To run the full suite in a generated project's CUDA-enabled environment,
first create and prepare that project from the repository root. Set the
cuBLAS workspace before Python starts: earlier tests can initialise CUDA
before later tests enable deterministic execution.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
uv run bokete init my-experiment
uv run --project my-experiment --with-editable . python -m unittest discover -s tests -q
```
