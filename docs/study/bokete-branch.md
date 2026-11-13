# All His Engines: BOKeTE Branch Study Guide

## Snapshot and scope

| Item | Recorded value |
| :--- | :--- |
| Mode | Changes against the shared ancestor with the default branch, master |
| Target branch | feat/project-template-and-cli-init |
| Target commit | 3e52760c93c7d1631200aa02a73e4f02a2c6555e |
| Base commit | 376fc3350451f62dc6eeaffc2a808507a8a816f1 |
| Inspected | 11 November 2026, Asia/Kuala_Lumpur, UTC+08:00 |
| Working tree before writing | Clean; no tracked modifications or untracked files |
| Intended output | docs/study/bokete-branch.md |
| Prerequisites | Basic Python functions, dictionaries, dataclasses and PyTorch model/optimiser concepts |

On disagreement, the code wins, then the repository documentation, then this guide. Identifiers keep their source spelling; the prose uses British English.

This branch turns basic training helpers into a project-generation and experiment workflow with independent trials, parameter sweeps, consistent validation selection, resumable state and reports. The name mentions a CLI, but the bloody engine room came along too.

The base already had a Trainer, early stopping, checkpoint saves and experiment helpers. This is a guide to how the branch extends and connects them, not a claim that it invented training. The old experiment code can be read with the historical command below.

## How to use this

This guide describes the recorded commit, before the starter moved to src/bokete/templates/starter/. Its source links open that historical snapshot; use the [getting-started guide](../getting-started.md#running-the-included-starter-template) for the current project-generation workflow.

Open this file beside the source in VS Code or Cursor. Follow the six sessions in order. Read each linked definition before reading its explanation, then answer the folded questions without peeking. A function name is a door handle, not evidence that you know what lives behind it.

Line links were generated from distinctive source text and are pinned to the snapshot above. To inspect that code locally, use a checkout of the recorded commit and the labelled definition or search command below.

Run these from the repository root in PowerShell. These exact read-only commands were checked while preparing the guide:

~~~powershell
git diff --name-status 376fc3350451f62dc6eeaffc2a808507a8a816f1
git diff 376fc3350451f62dc6eeaffc2a808507a8a816f1 -- src/bokete/experiments.py
git show 376fc3350451f62dc6eeaffc2a808507a8a816f1:src/bokete/experiments.py
rg -n 'def create_trial_runner|def fit|def selection_summary' src/bokete
Get-Content example/configs/example.yaml
~~~

The fixed hash keeps the comparison tied to this guide even if master advances. The path after the double hyphen limits the diff to one file. The colon in the git show argument selects a file at the old commit. rg searches definitions without importing the package or starting training.

All session exercises below are paper traces or source inspections. They do not require altering files or running training. No tests, builds, scaffolding commands or MNIST downloads were run for this guide. Test functions are evidence of what is asserted in source, not fresh test outcomes.

The README owns installation and API recipes. Start at [`usage documentation`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/README.md#L96) when you need a runnable example rather than an explanation.

## The whole picture

The branch separates deciding which experiment to run from executing one trial, and separates model selection from optional test evaluation.

~~~text
Installation / packaging
  pyproject.toml -> bokete.cli:main -> init_project -> copied starter

A starter experiment
  example/main.py
    -> parse_cli_config -> YAML -> nested dataclasses -> asdict
    -> create_trial_runner(factories, optional test callback)
    -> run_experiments
         -> parameter combinations, if a nonempty grid exists
         -> _run_trials
              -> runner(trial number, configuration)
                   -> seed and policy
                   -> loaders / model / criterion / optimiser / scheduler
                   -> Trainer.fit
                        -> train epoch -> validation
                        -> selection -> other callbacks -> checkpoint callbacks
                   -> optional final and best test evaluations
                   -> trial_report -> PNG and Markdown
              -> config_report
         -> experiment_report for a sweep

Interruption
  Trainer.fit -> TrainingInterrupted(history)
    -> runner writes partial report and re-raises
    -> _run_trials retains a status record and stops
    -> sweep retains its current configuration and stops

Resume
  explicit configuration/trial path
    -> reconstruct matching objects
    -> restore saved state and RNGs
    -> run only epochs after the saved boundary
~~~

The optional test callback runs after training. It does not choose the best epoch. That distinction is the ward keeping evaluation from changing the selection contract.

## Session 1: How the project becomes a command and a starter

### Goal

Explain how the installed command reaches Python, why the starter files must be packaged, and how a generated project stops depending on this checkout's parent directory.

### Read in this order

1. [`command registration`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/pyproject.toml#L35) and [`template packaging manifest`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/pyproject.toml#L46).
2. [`CLI argument parsing`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/cli.py#L63) followed by [`project generation`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/cli.py#L18).
3. [`public training imports`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/__init__.py#L7) and [`package version`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/__init__.py#L51).
4. [`starter dependency sources`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/example/pyproject.toml#L24).
5. [`portable scaffold regression`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/tests/test_regressions.py#L259) and [`CLI failure tests`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/tests/test_cli.py#L8).

### How it works

The script registration maps the command name bokete to the main function in bokete.cli. Installing the package establishes that link. Having a cli.py file in a checkout does not itself create the installed shell command.

The parser accepts init and a directory name. init_project refuses an existing target, then looks for the template inside the installed package. If that directory is absent, it looks in the source checkout. These are two locations for the same template, not two different project formats.

Packaging explicitly includes the starter files. The generator copies a named list rather than walking the whole example directory. That keeps a developer's environment, downloaded data and old results out of somebody else's new project. Copying a whole folder would be like handing Chas your coat with the cursed receipts still in it.

For a hypothetical init study-demo, trace the intended files into study-demo, then follow the text replacements in its pyproject.toml. The generated name becomes the destination basename, the BOKeTE requirement gains a minimum version, and the editable parent-directory source is removed. The CUDA package index remains part of the starter configuration.

This walkthrough did not execute the command. The regression constructs a fake template and checks the resulting file set and rewritten dependency text. It is not an installed-wheel integration test.

### Try it

Compare the template_files tuple with the wheel's force-include entries. Every copied file needs a packaged source. Predict the failure when just one individual template file is missing: there is no transaction that rolls back earlier copies.

### Questions

<details><summary><b>Q1.</b> Why does the generator need both a packaging manifest and a copy list?</summary>

The manifest makes the files available in an installed wheel. The copy list chooses which available files enter a new project. One controls distribution; the other controls generation. A correct copy list cannot recover a file omitted from the wheel.

</details>

<details><summary><b>Q2.</b> Why remove the editable parent-directory source from the generated project?</summary>

That source is for developing the example beside this checkout. In a standalone generated project its parent need not contain BOKeTE. Removing it lets the rewritten package requirement resolve independently of that workspace layout.

</details>

## Session 2: Turn configuration into an independent trial

### Goal

Trace the supplied example from command-line input to a built model and explain why factories run again for every trial.

### Read in this order

1. [`example entry point`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/example/main.py#L19) and [`configuration parsing`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/utils.py#L612).
2. [`path resolution`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/utils.py#L279) and [`dataclass loading`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/utils.py#L580).
3. [`root schema`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/example/src/schema.py#L31), [`training schema`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/example/src/schema.py#L18) and [`active grid`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/example/configs/example.yaml#L7).
4. [`runner construction`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/experiments.py#L301), then its inner run_fn.
5. [`starter model`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/example/src/models.py#L5) and [`trial runner test`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/tests/test_experiments.py#L106).

### How it works

The example parses --config into ExperimentConfig, including nested dataclasses. It then converts the result to a dictionary before handing it to the experiment layer. That conversion lets factories use ordinary dictionary lookups.

resolve_config_path distinguishes an explicit path from a short name such as example. load_config reads the resolved YAML. load_config_as fills fields declared by the schema, recurses into nested dataclasses and leaves omitted values at their dataclass defaults. It preserves an explicit null, but it is not a runtime type validator and does not reject unknown keys.

create_trial_runner returns a function that remembers the supplied factories. Building the runner does not train anything. Calling the returned function seeds the trial, obtains its resume paths, then constructs loaders, model, criterion, optimiser and optional scheduler. A new call builds new objects, so trial 2 is not a continuation of trial 1.

Worked trace: use the first combination in the supplied YAML and trial 2. With base seed 42, the runner selects seed 43. The example's data factory still uses a separate generator seeded with 42 for the training/validation split, so the split stays fixed while the trial's stochastic training can differ. The selected hidden dimension reaches SimpleMLP through model_factory. The scheduler factory, if supplied, receives the newly built optimiser and the same configuration.

The runner defaults to CrossEntropyLoss and Adam unless configuration or factories replace them. Its lr fallback is 0.001. The scheduler factory defaults to absent; there is no general YAML scheduler-name loader hiding elsewhere.

Notice the order: loaders before model, and optimiser before scheduler. Factories can consume random numbers, which is why seeding must happen before them. Report metadata comes from the actual model rather than constructing a second model just for its description.

### Try it

On paper, add training.amp: true to the YAML and follow the declared training schema. The snapshot schema has no amp field, so loading through that schema discards it. Contrast that with a direct dictionary passed to the runner, where the amp lookup exists. This is an inspection exercise, not a request to edit the YAML.

### Questions

<details><summary><b>Q1.</b> Why would constructing the optimiser once outside the trial function change the meaning of multiple trials?</summary>

The optimiser owns accumulated state and references model parameters. Reusing it can carry training history or parameter references across trials. The runner constructs it from each trial's newly constructed model.

</details>

<details><summary><b>Q2.</b> Does changing the root seed change the example's training/validation partition?</summary>

Not through that setting alone. The example's random_split receives its own generator explicitly seeded with 42. The trial seed affects global RNGs, while this independently supplied split generator uses its explicit seed.

</details>

## Session 3: One epoch, one selection rule

### Goal

Explain how a batch becomes an epoch loss, how validation accuracy is collected, and why minimum loss, selected epoch and final epoch can disagree.

### Read in this order

1. [`trainer construction`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/training.py#L330) and its _train_epoch method.
2. [`loss denominator`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/training.py#L65), [`loss contribution`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/training.py#L90) and [`epoch finalisation`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/training.py#L95).
3. [`validation pass`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/training.py#L264).
4. [`history record`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/metrics.py#L19), [`monitor direction`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/metrics.py#L46) and [`strict improvement`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/metrics.py#L54).
5. [`fit loop`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/training.py#L473) and [`accuracy selection regression`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/tests/test_training.py#L252).
6. [`uneven batch regression`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/tests/test_regressions.py#L72).

### How it works

Trainer holds a model, a loss function and an optimiser. Each training batch is prepared on the chosen device, ignored-only batches are skipped, gradients are cleared, the forward pass computes a scalar loss, and the scaler drives backward and the optimiser step. AMP is active only when requested on CUDA. If gradient clipping is requested through Trainer, gradients are unscaled first so the clipping threshold applies to actual gradients.

A batch mean is not an epoch mean when batches have different sizes. For mean-reduced MSE, imagine two scalar observations with batch mean 1 and one observation with mean 9. Pooling gives (2 * 1 + 1 * 9) / 3 = 11 / 3. Averaging the two batch means gives 5 and overweights the shorter batch. These numbers are a paper example of the code's arithmetic, not measurements from training.

The denominator depends on the criterion. Elementwise losses count elements; class-index cross-entropy can count valid targets or class-weight sums; batchmean uses batch size; sum stays a sum. Custom scalar criteria are treated as per-sample means where the code can infer a batch size. Follow the helper branches rather than applying one universal denominator.

Validation runs without gradient tracking. When accuracy is the monitor, it collects valid-target top-1 accuracy in that same pass. When loss is the monitor, accuracy is not automatically recorded. The README owns the exact supported target shapes and ignored-target rules at [`validation selection reference`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/README.md#L300).

Worked selection trace, expressed as hypothetical observed scores:

| Epoch | Validation loss | Validation accuracy |
| :---: | :---: | :---: |
| 1 | 0.10 | 0.60 |
| 2 | 0.50 | 0.90 |
| 3 | 0.40 | 0.90 |

Monitoring accuracy selects epoch 2. Its selected validation loss is 0.50, the minimum validation loss is 0.10, and the final validation loss is 0.40. The tie at epoch 3 does not replace epoch 2. Monitoring loss would instead select epoch 1.

The same monitor and inferred direction reach the built-in checkpoint and early-stopping callbacks. Early stopping additionally uses min_delta: a small strict improvement can save a new best checkpoint without resetting its patience. The checkpoint and stopping policies are related, not identical twins. Trust neither just because they wear matching coats.

### Try it

Using the table, work out the selected epoch for each monitor, then compare your reasoning with selection_summary. Read the clipping order and explain why moving unscale_ after clipping would change what max_grad_norm means. Do not move it.

### Questions

<details><summary><b>Q1.</b> Why can a new best checkpoint be saved while early stopping still accumulates a bad epoch?</summary>

Checkpoint selection uses strict improvement with no minimum margin. EarlyStopping compares against its own best score using min_delta. An improvement smaller than that margin can satisfy the first policy and fail the second.

</details>

<details><summary><b>Q2.</b> Why must reports retain both selected and final values when monitoring accuracy?</summary>

The final epoch need not have the highest accuracy, and the accuracy-selected epoch need not have the minimum loss. Labelling all three as best would erase which weights and which observation each number describes.

</details>

## Session 4: Save state, survive interruption, resume the right trial

### Goal

Explain what a resumable checkpoint contains, why epoch-end order matters, and how an interrupted trial remains visible without training the next one.

### Read in this order

1. [`checkpoint callback`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/training.py#L197) and [`atomic file replacement`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/training.py#L44).
2. The resume branch and checkpoint-state assembly inside fit from Session 3.
3. [`RNG capture`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/training.py#L113) and [`RNG restoration`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/training.py#L127).
4. [`interruption exception`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/training.py#L36) and [`trial orchestration`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/experiments.py#L30).
5. [`per-trial resume lookup`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/experiments.py#L475) and [`original-best retention`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/experiments.py#L485).
6. [`interrupted second-trial trace test`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/tests/test_roadmap.py#L47), [`resume equivalence test helper`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/tests/test_training.py#L48) and [`failed save regression`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/tests/test_regressions.py#L52).

### How it works

A resumable run needs more than weights. fit includes optimiser state, scaler state, stateful callback types and states, history, policy information and RNG state in a checkpoint. A scheduler participates through the callback state mechanism. Reconstruct the corresponding objects before loading their state.

The loop performs other epoch-end callbacks before assembling the checkpoint. That saves the scheduler and patience state after the completed epoch's updates. Saving first could resurrect a model at epoch 10 with its scheduler still behaving as though epoch 9 had just ended.

Each checkpoint file uses a sibling temporary file and replacement. The destination survives a failed serialisation, but last.pt and best.pt are separate writes. The pair is not one transaction. The README owns recovery requirements and unsupported RNG/sampler state at [`checkpoint resume reference`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/README.md#L338).

Worked interruption trace: request three trials, each targeting 20 epochs. Trial 1 reaches 20. Trial 2 saves epoch 7, then Ctrl+C arrives during validation for epoch 8. The training-loss append happens after validation, so that unfinished pair is absent. TrainingInterrupted carries the retained history; the runner writes a partial report; _run_trials returns trial 1 and an interrupted trial 2 record, then stops before trial 3.

For the next invocation, explicitly map trial 1 to its own epoch-20 checkpoint and trial 2 to its own epoch-7 checkpoint. Trial 1 returns saved history without new epochs; trial 2 starts at epoch 8; an unlisted trial 3 starts fresh. This example assumes those checkpoints and matching objects exist. It is not automatic discovery of the previous run.

RNG restoration happens after reconstructing objects and loading state. Model construction itself consumes random numbers; restoring before construction would let that setup shift the resumed random sequence.

A further distinction matters for reports. last.pt can contain later weights while the selected best lives in an older best.pt. The runner retains a matching original best file into the new checkpoint directory when saving is enabled. Missing original best weights are not invented from the final model. If writes are disabled and a resumed run improves beyond the retained best, the old weights are marked unavailable for the new selection. The regression at [`disabled-write best-weight case`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/tests/test_roadmap.py#L189) checks that particular trap.

### Try it

Draw four boxes labelled current model, history, last.pt and best.pt for the interruption case. Mark which completed epoch each represents. Then trace a resume whose target is already reached. Keep the paper hypothetical; it does not overwrite any checkpoints.

### Questions

<details><summary><b>Q1.</b> Why restore RNG state after constructing the new model and loaders?</summary>

Those constructors may consume random numbers. Restoring last makes the next training operation continue from the saved random sequence rather than the sequence left after reconstruction.

</details>

<details><summary><b>Q2.</b> Why is retaining best.pt necessary when resuming from last.pt into a new directory?</summary>

The resumed selected epoch can be older than the last checkpoint's current weights. If no new improvement occurs, the new training invocation never writes a replacement best file. A report needs the original selected weights to evaluate best-weight test metrics.

</details>

<details><summary><b>Q3.</b> Does an interrupted history prove the latest epoch was saved?</summary>

No. Complete training/validation pairs can be recorded before callbacks or checkpoint writes finish. The latest loadable checkpoint is the resume boundary; history can describe work that has to be repeated after that boundary.

</details>

## Session 5: Sweeps, reports and the meaning of an average

### Goal

Follow a configuration grid into independent trials and explain which observations contribute to each report.

### Read in this order

1. [`experiment dispatch`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/experiments.py#L144) and its grid/resume-map branches.
2. [`run directory allocation`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/utils.py#L490) and [`nested key writes`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/utils.py#L569).
3. [`selection from recorded history`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/metrics.py#L61) and [`history summary`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/metrics.py#L80).
4. [`trial report`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/reporting.py#L152), [`configuration report`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/reporting.py#L349) and [`sweep report`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/reporting.py#L483).
5. [`individual loss plot`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/plotting.py#L33) and [`multi-trial plot`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/plotting.py#L87).
6. [`selection-aware reporting tests`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/tests/test_reporting.py#L91), [`sweep interruption test`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/tests/test_roadmap.py#L75) and [`plot tests`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/tests/test_plotting.py#L13).

### How it works

run_experiments dispatches to one configuration's trials or a Cartesian product of grid values. It deep-copies configuration before applying dotted-key overrides so one combination cannot leak changes into the next.

The supplied example has two learning rates and two hidden dimensions, in that key order. On paper the combinations are (0.001, 64), (0.001, 128), (0.0005, 64) and (0.0005, 128). With three trials each, that requests 12 trials if nothing interrupts or stops the orchestration. These counts come from the YAML and product rule; they are not a claim that this guide ran MNIST.

The example's first combination has the folder prefix 001 and parameter suffix lr_0.001_hidden_dim_64. The numeric prefix matters because sanitised parameter labels can collide. Run-directory allocation also retries an exclusive directory creation when another run has claimed its candidate name.

Sweeps use configuration:trial resume keys. The hypothetical key "2:1" selects combination 2, trial 1. The orchestration reduces that mapping to trial-only keys before calling the runner. Reordering grid keys or values changes those numeric addresses, which is why paths still need human care.

The output has three scopes: trial_report describes one history; config_report aggregates completed trials of one configuration; experiment_report compares configuration summaries. They select from recorded values rather than running validation again.

Worked ranking trace: configuration A has completed trials with selected losses 0.5 and 0.7. Its mean selected loss is 0.6. Configuration B has one completed selected loss 0.8 and an interrupted trial with partial selected loss 0.1. B's ranking score is 0.8 because the interrupted observation is excluded. Including 0.1 would reward the unfinished trial. No occult interpretation makes that fair.

Optional test metrics are evaluated twice only when matching best weights are available: first using the final in-memory model, then using selected best weights. They receive Final and Best prefixes. Reporting may average numeric extra values while leaving nonnumeric values visible without numerical aggregation. Test callbacks do not alter the training monitor.

Mean loss curves use only completed paired histories supplied by config_report, and truncate to the shortest paired history. Later epochs from longer trials therefore do not appear in that shared mean plot. The raw trial histories still retain them.

### Try it

Find the path that appends the current sweep configuration after a trial is interrupted. Then read the report filtering separately. Explain how an interrupted configuration can still have a ranking based on its earlier completed trials. The exclusion applies to interrupted trials, not automatically to every completed sibling.

### Questions

<details><summary><b>Q1.</b> Why is Mean Best Validation Loss different from taking the minimum of the mean loss curve?</summary>

The report selects each completed trial's best score first and averages those scores. The plot averages losses at shared epoch positions. Different trials can select different epochs, so those are different operations.

</details>

<details><summary><b>Q2.</b> Why can a sweep configuration labelled interrupted still have a numerical rank?</summary>

Its status records that orchestration stopped during that configuration. Its completed trial histories can still contribute to its score. Only interrupted trial records are filtered from those aggregates.

</details>

## Session 6: Reproducibility is a policy, not a seed incantation

### Goal

Distinguish hardware availability from the installed PyTorch build, understand process-wide settings, and judge what a regression test actually proves.

### Read in this order

1. [`seeding and deterministic policy`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/utils.py#L155), [`runtime capture`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/utils.py#L182) and [`device selection`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/utils.py#L260).
2. The deterministic guard in _train_epoch and policy checks in fit from earlier sessions.
3. [`recorded runtime formatting`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/reporting.py#L64).
4. [`example CUDA index`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/example/pyproject.toml#L20) and [`documented environment and verification`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/README.md#L465).
5. [`CUDA resume regression`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/tests/test_training.py#L125), [`policy transition test`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/tests/test_roadmap.py#L251) and [`runtime reporting test`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/tests/test_roadmap.py#L341).
6. [`example import test`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/tests/test_experiments.py#L91) and [`utility tests`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/tests/test_utils.py#L35).

### How it works

set_seed seeds global Python, NumPy and PyTorch generators and explicitly switches deterministic algorithm policy on or off. It also sets cuDNN's deterministic flag, resets benchmarking and supplies a missing cuBLAS workspace setting when deterministic mode is requested. That setting must precede relevant CUDA initialisation. It cannot retroactively rebuild a context that already chose its configuration.

During training, the deterministic guard prevents automatic benchmarking from re-enabling itself against that policy. With restrictions disabled, consistent input shapes can enable benchmarking; a detected shape change disables it. These are process-wide backend settings, not private properties of one Trainer object.

The report's runtime table comes from history captured during execution. A report written later does not claim that the report writer's current environment was the original training environment. A resumed invocation records its current runtime while retaining earlier metric history. Full-state policy checks compare recorded algorithm settings before continuing, but do not validate every property of the data pipeline.

The README records a CPU-only root environment and a CUDA-enabled example environment. Those are installations, not competing claims about whether a physical GPU exists. The README also records RTX 3060 CUDA resume verification. That is prior verification recorded by the repo; this guide did not repeat it or re-detect installed hardware.

Read check_resume_matches_uninterrupted as an argument. It trains a reference run, trains a shorter split run, reconstructs objects under a different seed, resumes, and compares model, optimiser, scheduler, stopper, scaler, metric and random-generator state. The CUDA wrapper adds a hardware-availability skip guard and enables AMP. It is a controlled equivalence regression, not proof that arbitrary datasets, workers or distributed jobs resume identically.

Finally, test names are not results. test_example_run calls runpy.run_path without giving the script the __main__ name. The example's guarded main function is therefore not invoked by that test. It imports the example and checks the returned namespace; it does not establish a full MNIST training run. Read the assertions before believing the sermon.

### Try it

Inspect the CUDA skip decorator, then the comparison helper. List two states it compares and two data-pipeline states it never captures. For an optional future runtime check, use the command already maintained in the README. It was not run for this guide; it runs tests that create temporary files and some ignored results. The default example training command can download MNIST and create data/results, so it is not a read-only inspection.

### Questions

<details><summary><b>Q1.</b> Why can a machine with an NVIDIA GPU still skip the CUDA test?</summary>

The guard checks torch.cuda.is_available in the Python process running the tests. A CPU-only PyTorch installation cannot expose that GPU through CUDA. The example dependency source can provide a different build from the root environment.

</details>

<details><summary><b>Q2.</b> Why is the resume regression's deliberately different reconstruction seed useful?</summary>

It makes reconstruction drift visible. A resume implementation that restores weights but forgets RNG state could look plausible under a convenient seed, yet fail the comparison against uninterrupted training and the next sampled random values.

</details>

## Where the ice is thin

These findings come from reading this snapshot, not new runtime experiments. They are ordered by the chance of losing intended behaviour or obscuring evidence. They are study notes, not fixes made by this guide.

1. **Configuration can silently lose intent.** The schema loader visits declared dataclass fields and ignores unknown keys. TrainingConfig lacks amp and max_grad_norm; the root schema lacks device. Separately, the standard runner auto-detects device and does not forward max_grad_norm to Trainer. Adding a YAML setting is therefore not enough to ensure the engine consumes it. Follow [`schema field filtering`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/utils.py#L596), [`starter training fields`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/example/src/schema.py#L18) and [`runner device selection`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/experiments.py#L378). Type annotations alone do not validate incoming values either.

2. **Logging cleanup reaches outside the experiment.** [`root file-handler cleanup`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/utils.py#L481) closes every FileHandler attached to Python's root logger. _run_trials invokes it in finally. A host application's own file handler can therefore be detached along with BOKeTE's handler. [`logging setup`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/utils.py#L416) also configures the root logger, so this integration boundary deserves attention.

3. **Resume paths remain a provenance contract.** Original-best retention compares epoch, score and monitor, not a unique trial identity. Two unrelated checkpoints can share those values. Sweep resume addresses also depend on grid order. The code does not prove that the supplied path belongs to the intended experiment. This limit is admitted in [`runner resume instructions`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/README.md#L385).

4. **A report can outlive a failed plot.** [`plot finalisation`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/plotting.py#L17) logs exceptions from saving the image and does not re-raise them. Report generation can continue and emit an image reference for a missing file. This is not evidence that a failure occurred here; it is the failure path visible in the source.

5. **Scaffolding can leave a partial target.** init_project rejects an existing directory, but its copy loop has no rollback if a later copy fails. The next identical invocation then meets an existing target. The file-list regression checks portable contents; it does not establish atomic project creation or a built wheel's installed behaviour.

6. **The starter's comments describe the wrong default.** [`starter sweep comment`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/example/configs/example.yaml#L6) and the README's starter instructions say to uncomment the grid, but the grid is already enabled. Following the supplied example requests the combinations described in Session 5 rather than a single configuration.

7. **The example import test is narrower than its name.** The runpy call discussed in Session 6 does not invoke guarded main. Treat that test as import coverage, not a fresh end-to-end MNIST result.

The repository already admits broader resume limits at the checkpoint reference: no mid-batch continuation, no independently owned RNG restoration, no persistent-worker or custom sampler/dataset state capture, and no distributed per-rank state capture. It also documents independent checkpoint writes and process-wide deterministic settings. Use those README sections as the maintained reference instead of treating this guide as a warranty from a bloke with a lighter.

## Change index and reading map

This index accounts for every path changed against the base, including the deleted test module. It groups files by the first session that needs them; shared core modules return in later sessions.

A denotes a file added by the branch, M an existing file changed, and D a deleted file. The lockfiles are generated dependency data: keep them in the index, skip their individual package entries during the reading sessions.

### Session 1: Distribution and project generation

| Status | Path | Reading purpose |
| :---: | :--- | :--- |
| M | [`pyproject.toml`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/pyproject.toml#L1) | Package metadata, command registration and bundled template |
| M | [`src/bokete/__init__.py`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/__init__.py#L5) | Public imports and package version |
| A | [`src/bokete/cli.py`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/cli.py#L18) | Scaffold parser and generation |
| A | [`example/pyproject.toml`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/example/pyproject.toml#L1) | Starter dependencies and editable/CUDA sources |
| A | [`example/.gitignore`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/example/.gitignore#L1) | Generated project's local artefact exclusions |
| A | [`example/src/__init__.py`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/example/src/__init__.py#L2) | Starter source-package marker |
| A | [`tests/test_cli.py`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/tests/test_cli.py#L8) | CLI failure assertions |

### Session 2: Configuration and factories

| Status | Path | Reading purpose |
| :---: | :--- | :--- |
| A | [`example/main.py`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/example/main.py#L19) | Real starter flow and user-supplied factories |
| A | [`example/configs/example.yaml`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/example/configs/example.yaml#L1) | Values used in the worked sweep trace |
| A | [`example/src/schema.py`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/example/src/schema.py#L31) | Declared configuration boundary |
| A | [`example/src/models.py`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/example/src/models.py#L5) | Example model's input-to-output wiring |
| M | [`src/bokete/utils.py`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/utils.py#L580) | Configuration, policy, directory and logging support |
| M | [`src/bokete/experiments.py`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/experiments.py#L144) | Trial construction, orchestration and resume routing |
| A | [`tests/test_experiments.py`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/tests/test_experiments.py#L19) | Runner and sweep assertions |

### Sessions 3 and 4: Training, selection and resurrection

| Status | Path | Reading purpose |
| :---: | :--- | :--- |
| M | [`src/bokete/training.py`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/training.py#L330) | Training/validation, callbacks, checkpoints and resume |
| M | [`src/bokete/metrics.py`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/metrics.py#L19) | Histories, monitored selection and summaries |
| A | [`tests/test_training.py`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/tests/test_training.py#L27) | Resume equivalence and selection assertions |
| A | [`tests/test_regressions.py`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/tests/test_regressions.py#L43) | Loss, save, scaffold, path and reporting regression cases |
| A | [`tests/test_roadmap.py`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/tests/test_roadmap.py#L18) | Interrupted results, per-trial resume and deterministic policy |
| D | tests/test_bokete.py | Former combined test module; inspect its base version through git show |

### Session 5: Reports and plots

| Status | Path | Reading purpose |
| :---: | :--- | :--- |
| M | [`src/bokete/reporting.py`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/reporting.py#L152) | Trial, configuration and sweep report scopes |
| M | [`src/bokete/plotting.py`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/src/bokete/plotting.py#L33) | Individual and shared-epoch loss plots |
| A | [`tests/test_reporting.py`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/tests/test_reporting.py#L21) | Report labels, monitor consistency and extra metrics |
| A | [`tests/test_plotting.py`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/tests/test_plotting.py#L13) | Plot artefact assertions |

### Session 6 and supporting material

| Status | Path | Reading purpose |
| :---: | :--- | :--- |
| A | [`tests/test_utils.py`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/tests/test_utils.py#L35) | Configuration, directory and utility assertions |
| M | [`README.md`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/README.md#L3) | Maintained recipes, supported contracts and admitted limits |
| M | [`LICENSE`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/LICENSE#L6) | Changed permissions notice; read the notice itself for its terms |
| M | [`.gitignore`](https://github.com/junhaochai/BOKeTE/blob/3e52760c93c7d1631200aa02a73e4f02a2c6555e/.gitignore#L42) | Ignore rules for runtime outputs and development artefacts |
| M | uv.lock | Generated root dependency resolution; skip package-by-package reading |
| A | example/uv.lock | Generated example dependency resolution; skip package-by-package reading |

### Refresh the index

The first command under How to use this is the authoritative tracked-path list for this base and the current working tree. Run git ls-files --others --exclude-standard alongside it to catch new files that no diff can show. Both commands were checked during preparation.

The guide itself is a new untracked document, so it is not one of the branch's 30 changed paths at the recorded snapshot. No repository clean-tree check was found in the tracked files inspected. It still appears in git status until you decide what to do with it.

If you need the removed combined tests, this read-only command was also checked:

~~~powershell
git show 376fc3350451f62dc6eeaffc2a808507a8a816f1:tests/test_bokete.py
~~~

The guide follows the code's behaviour at its recorded commit. When the code moves, refresh the links and retrace the argument. Old wards are how you end up owing somebody with hooves.
