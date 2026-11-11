"""
Core training machinery for bokete.

Key Classes & Functions:
  - Trainer      : Main training loop engine handling fit(), epochs, AMP, and callback hooks.
  - EarlyStopping: Callback for halting training when the monitored validation metric stalls.
  - Checkpoint   : Callback for saving best.pt and last.pt model state dictionaries.
  - evaluate     : Computes evaluation loss over a DataLoader without tracking gradients.
"""

from datetime import datetime
import logging
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Callable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm

import bokete.metrics as metrics
import bokete.utils as utils





logger = logging.getLogger(__name__)


class TrainingInterrupted(KeyboardInterrupt):
    """Carry completed epoch history while preserving KeyboardInterrupt semantics."""

    def __init__(self, history):
        super().__init__("Training interrupted")
        self.history = history


def _atomic_save(state, destination):
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(state, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _valid_targets(criterion, targets):
    if not isinstance(targets, torch.Tensor):
        return None
    ignore_index = getattr(criterion, "ignore_index", None)
    if ignore_index is not None and not targets.is_floating_point():
        return targets != ignore_index
    return torch.ones_like(targets, dtype=torch.bool)


def _loss_weight(criterion, outputs, targets):
    """Denominator of a batch mean, used to pool scalar losses across an epoch."""
    reduction = getattr(criterion, "reduction", "mean")
    if reduction not in ("mean", "sum", "batchmean"):
        raise ValueError("Loss reporting requires a scalar mean, sum or batchmean reduction.")
    if reduction == "batchmean" or not isinstance(targets, torch.Tensor):
        return outputs.shape[0] if isinstance(outputs, torch.Tensor) and outputs.ndim else 1
    if isinstance(criterion, (torch.nn.CrossEntropyLoss, torch.nn.NLLLoss)):
        if targets.is_floating_point():
            class_dim = 1 if targets.ndim > 1 else 0
            return targets.numel() / targets.shape[class_dim]
        valid_targets = targets[_valid_targets(criterion, targets)]
        if reduction == "mean" and criterion.weight is not None:
            return criterion.weight[valid_targets.long()].sum().item()
        return valid_targets.numel()
    if isinstance(criterion, (
        torch.nn.L1Loss, torch.nn.MSELoss, torch.nn.SmoothL1Loss, torch.nn.HuberLoss,
        torch.nn.BCELoss, torch.nn.BCEWithLogitsLoss, torch.nn.KLDivLoss,
        torch.nn.PoissonNLLLoss, torch.nn.SoftMarginLoss, torch.nn.HingeEmbeddingLoss,
    )):
        return math.prod(torch.broadcast_shapes(outputs.shape, targets.shape))
    # Custom scalar criteria are assumed to return a per-sample mean.
    return targets.shape[0] if targets.ndim else 1


def _loss_contribution(value, weight, reduction):
    """Convert a batch scalar to its contribution to the epoch total."""
    return value if reduction == "sum" else value * weight


def _epoch_loss(total, weight, reduction):
    """Finalise a loss total once the caller has checked its denominator."""
    return total if reduction == "sum" else total / weight


def _loader_generators(loaders):
    generators = {}
    for name, loader in zip(("train", "val"), loaders):
        for source, owner in (
            ("loader", loader),
            ("sampler", getattr(loader, "sampler", None)),
        ):
            generator = getattr(owner, "generator", None)
            if isinstance(generator, torch.Generator):
                generators[f"{name}.{source}"] = generator
    return generators


def _capture_rng_state(loaders):
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (name, keys.tolist(), position, has_gauss, cached_gaussian),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loaders": {
            name: generator.get_state()
            for name, generator in _loader_generators(loaders).items()
        },
    }


def _restore_rng_state(state, loaders):
    generators = _loader_generators(loaders)
    if generators.keys() != state["loaders"].keys():
        raise ValueError("Resume requires matching DataLoader generator configuration.")
    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(state["cuda"]) != cuda_count:
        raise ValueError("Resume requires the same number of visible CUDA devices.")

    random.setstate(state["python"])
    name, keys, position, has_gauss, cached_gaussian = state["numpy"]
    np.random.set_state((
        name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached_gaussian,
    ))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    for name, generator in generators.items():
        generator.set_state(state["loaders"][name])


class EarlyStopping:
    """
    Signals that training should stop once the monitored validation metric has not improved
    by more than min_delta for `patience` consecutive epochs.
    """

    def __init__(self, patience=10, min_delta=0.0, *, monitor="val_loss"):
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative.")
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = metrics.monitor_mode(monitor)
        self.best_score = None
        self.bad_epochs = 0

    def step(self, val_loss):
        """Record the monitored score. Returns True when patience is exhausted."""
        if self.patience is None or self.patience <= 0:
            return False
        if metrics.is_improvement(val_loss, self.best_score, self.mode, self.min_delta):
            self.best_score = val_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def state_dict(self):
        """Return the stopping policy and accumulated patience state."""
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_score": self.best_score,
            "bad_epochs": self.bad_epochs,
        }

    def load_state_dict(self, state):
        """Restore the stopping policy and accumulated patience state."""
        self.patience = state["patience"]
        self.min_delta = state["min_delta"]
        self.best_score = state["best_score"]
        self.bad_epochs = state["bad_epochs"]

    def on_epoch_end(self, trainer: Any, epoch: int, val_loss: float) -> None:
        """Callback hook for Trainer fit loop."""
        score = val_loss if self.monitor == "val_loss" else trainer.current_metrics[self.monitor]
        if self.step(score):
            trainer.should_stop = True


class Checkpoint:
    """
    Saves model weights during training: 'last.pt' every epoch and 'best.pt'
    whenever the monitored validation metric improves. Files contain the epoch,
    validation loss, selection metadata and model state_dict. Trainer checkpoints
    also contain the optimiser, scheduler, scaler, history and RNG state needed
    to resume at a completed epoch boundary.
    """

    def __init__(self, directory, save_best=True, save_last=True, enabled=True, *, monitor="val_loss"):
        self.directory = Path(directory)
        self.save_best = save_best
        self.save_last = save_last
        self.enabled = enabled
        self.monitor = monitor
        self.mode = metrics.monitor_mode(monitor)
        self.best_score = None
        self.best_saved = False

    def update(self, model, epoch, val_loss, optimizer=None, *, score=None, training_state=None):
        if not self.enabled or not (self.save_best or self.save_last):
            return
        if score is None:
            if self.monitor != "val_loss":
                raise ValueError("An explicit score is required when monitoring validation accuracy.")
            score = val_loss
        improved = metrics.is_improvement(score, self.best_score, self.mode)
        best_score = score if improved else self.best_score
        self.directory.mkdir(parents=True, exist_ok=True)
        # Extract inner model from multi-GPU wrapper (DataParallel/DDP), or fallback to raw model on single-GPU/CPU
        raw_model = getattr(model, 'module', model)
        state = {
            'epoch': epoch,
            'val_loss': val_loss,
            'monitor': self.monitor,
            'mode': self.mode,
            'score': score,
            'best_score': best_score,
            'model_state_dict': raw_model.state_dict(),
        }
        if optimizer is not None:
            state['optimizer_state_dict'] = optimizer.state_dict()
        if training_state is not None:
            state['training_state'] = training_state

        if self.save_last:
            _atomic_save(state, self.directory / 'last.pt')
        if self.save_best and improved:
            _atomic_save(state, self.directory / 'best.pt')
            self.best_saved = True
        self.best_score = best_score

    def on_epoch_end(self, trainer: Any, epoch: int, val_loss: float) -> None:
        """Callback hook for Trainer fit loop."""
        score = val_loss if self.monitor == "val_loss" else trainer.current_metrics[self.monitor]
        self.update(
            trainer.model, epoch, val_loss, optimizer=trainer.optimizer, score=score,
            training_state=trainer._checkpoint_state,
        )


def _default_prepare_batch(batch, device):
    """Default batch handling: assumes (inputs, targets) pairs."""
    inputs, targets = batch
    return inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)


def evaluate(model, loader, criterion, device=None, prepare_batch=None, amp_active=False, *, collect_accuracy=False):
    """Compute pooled epoch loss, and optionally top-1 accuracy, in one pass.

    Returns a float by default, or a val_loss/val_accuracy dict when
    collect_accuracy=True. Accuracy requires multiclass scores and integer
    class-index targets; it is weighted by valid samples, including a short last
    batch. Targets matching criterion.ignore_index are excluded. Batches with
    no valid targets are skipped; validation with no valid targets is rejected.
    Mean losses use their sample/element or class-weight denominator; sum
    reduction remains a sum. Custom scalar criteria are treated as per-sample
    means. The same loss aggregation applies when accuracy is not collected.
    """
    device = torch.device(device) if device is not None else utils.determine_device()

    if prepare_batch is None:
        prepare_batch = _default_prepare_batch

    model.eval()
    running_loss = 0.0
    total_weight = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for batch in loader:
            inputs, targets = prepare_batch(batch, device)
            valid = _valid_targets(criterion, targets)
            if valid is not None and not valid.any():
                continue
            with torch.autocast(device_type=device.type, enabled=amp_active):
                outputs = model(inputs)
                weight = _loss_weight(criterion, outputs, targets)
                if weight == 0:
                    # A zero-weight class can still contribute label-smoothing
                    # loss, and its samples still count towards accuracy.
                    batch_total = (
                        torch.nn.functional.cross_entropy(
                            outputs, targets, weight=criterion.weight,
                            ignore_index=criterion.ignore_index, reduction="sum",
                            label_smoothing=criterion.label_smoothing,
                        ).item() if isinstance(criterion, torch.nn.CrossEntropyLoss) else 0.0
                    )
                else:
                    batch_loss = criterion(outputs, targets).item()
                    batch_total = _loss_contribution(batch_loss, weight, getattr(criterion, "reduction", "mean"))
            running_loss += batch_total
            total_weight += weight
            if collect_accuracy:
                if (not isinstance(targets, torch.Tensor)
                        or outputs.ndim != 2 or outputs.shape[1] < 2 or targets.ndim != 1
                        or outputs.shape[0] != targets.shape[0]
                        or targets.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
                    raise ValueError("val_accuracy requires [batch, classes] outputs and [batch] class-index targets.")
                correct += ((outputs.argmax(dim=1) == targets) & valid).sum().item()
                total += valid.sum().item()

    if total_weight == 0:
        raise ValueError("Cannot evaluate an empty loader or with no valid targets.")
    loss = _epoch_loss(running_loss, total_weight, getattr(criterion, "reduction", "mean"))
    if collect_accuracy:
        if total == 0:
            raise ValueError("Cannot monitor validation accuracy on an empty loader or with no valid targets.")
        return {"val_loss": loss, "val_accuracy": correct / total}
    return loss


class Trainer:
    """Handles the training and validation execution for a PyTorch model.

    Attributes:
        model (torch.nn.Module): The model to train.
        criterion (callable): Loss function applied as `criterion(outputs, targets)`.
        optimizer (torch.optim.Optimizer): The optimizer for updating weights.
        device (torch.device): Device on which to run the training.
        prepare_batch (callable): Callable `(batch, device) -> (inputs, targets)`.
        max_grad_norm (float, optional): Gradient clipping threshold (L2 norm).
        amp_active (bool): Whether mixed-precision (autocast) is active.
        scaler (torch.cuda.amp.GradScaler): Gradient scaler for AMP.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: Callable,
        optimizer: torch.optim.Optimizer,
        *,
        device: Optional[Any] = None,
        prepare_batch: Optional[Callable] = None,
        max_grad_norm: Optional[float] = None,
        amp: bool = False,
        log_model_structure: bool = False,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device(device) if device is not None else utils.determine_device()
        self.prepare_batch = prepare_batch or _default_prepare_batch
        self.max_grad_norm = max_grad_norm

        self.model.to(self.device)
        utils.log_model_info(self.model, log_layers=log_model_structure)

        # AMP only applies on CUDA; on CPU both autocast and the scaler become no-ops.
        self.amp_active = amp and self.device.type == 'cuda'
        scaler_type = getattr(torch.amp, "GradScaler", None)
        self.scaler = (
            scaler_type('cuda', enabled=self.amp_active) if scaler_type is not None
            else torch.cuda.amp.GradScaler(enabled=self.amp_active)
        )

        # Track spatial shape consistency across batches for safe cuDNN benchmarking
        self._last_spatial_shape = None
        self._variable_shapes_detected = False

        self.should_stop = False
        self.monitor = "val_loss"
        self.current_metrics = {}

    def evaluate(self, loader: DataLoader, *, collect_accuracy: bool = False):
        """Compute the mean loss of `model` over `loader` using Trainer configuration."""
        return evaluate(
            self.model,
            loader,
            self.criterion,
            device=self.device,
            prepare_batch=self.prepare_batch,
            amp_active=self.amp_active,
            collect_accuracy=collect_accuracy,
        )

    def _train_epoch(
        self,
        train_loader: DataLoader,
        max_batches: Optional[int] = None,
        epoch_idx: Optional[int] = None,
        total_epochs: Optional[int] = None,
        show_progress: bool = True,
    ) -> float:
        """Run one training epoch over `train_loader` and return the mean training loss."""
        self.model.train()
        deterministic = (torch.are_deterministic_algorithms_enabled()
                         or torch.backends.cudnn.deterministic)
        if deterministic:
            torch.backends.cudnn.benchmark = False
        running_loss = 0.0
        total_weight = 0.0
        batches_run = 0

        if epoch_idx and total_epochs:
            desc = f" {utils.format_epoch_label(epoch_idx, total_epochs)}"
        else:
            desc = " Training"
        batch_pbar = utils.create_progress_bar(
            train_loader,
            desc=desc,
            dynamic_ncols=True,
            leave=False,
            disable=not show_progress,
            mininterval=0.5,
        )

        for batch in batch_pbar:
            if max_batches is not None and batches_run >= max_batches:
                break

            inputs, targets = self.prepare_batch(batch, self.device)
            valid = _valid_targets(self.criterion, targets)
            if valid is not None and not valid.any():
                continue

            # Auto-detect spatial shape consistency for cuDNN benchmarking
            if self.device.type == 'cuda' and not deterministic and not self._variable_shapes_detected:
                spatial_shape = tuple(inputs.shape[1:])
                if self._last_spatial_shape is None:
                    self._last_spatial_shape = spatial_shape
                    torch.backends.cudnn.benchmark = True
                elif spatial_shape != self._last_spatial_shape:
                    self._variable_shapes_detected = True
                    torch.backends.cudnn.benchmark = False

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, enabled=self.amp_active):
                outputs = self.model(inputs)
                weight = _loss_weight(self.criterion, outputs, targets)
                if weight == 0:
                    raise ValueError("Cannot train a mean-reduced batch with zero total class weight.")
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            if self.max_grad_norm is not None:
                # Gradients must be unscaled before clipping, otherwise the
                # threshold would apply to AMP-scaled values.
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_val = loss.item()
            running_loss += _loss_contribution(loss_val, weight, getattr(self.criterion, "reduction", "mean"))
            total_weight += weight
            batches_run += 1
            avg_loss = _epoch_loss(running_loss, total_weight, getattr(self.criterion, "reduction", "mean"))
            batch_pbar.set_postfix(loss=f"{avg_loss:.4f}", refresh=False)

        if total_weight == 0:
            raise ValueError("Cannot train on an empty loader or with no valid targets.")
        return round(avg_loss, 4)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        *,
        callbacks: Optional[List[Any]] = None,
        scheduler: Optional[Any] = None,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint: Optional[Checkpoint] = None,
        progress: bool = True,
        max_train_batches: Optional[int] = None,
        log_interval: int = 1,
        monitor: str = "val_loss",
        resume_from: Optional[Any] = None,
    ) -> metrics.TrainingMetrics:
        """Train while selecting epochs by val_loss (min) or val_accuracy (max).

        monitor is authoritative for built-in checkpoint and early-stopping
        callbacks. Custom on_epoch_end callbacks still receive validation loss;
        all collected values are available in trainer.current_metrics. A
        ReduceLROnPlateau scheduler receives the monitored score and must have
        the matching mode. Accuracy is collected only when it is monitored.

        resume_from is a full-state checkpoint path, or None for a fresh run.
        epochs is the total target, including previously completed epochs.
        Resume requires matching model/optimiser configuration, AMP setting,
        monitored metric and stateful callback types/order. History is retained;
        timing describes this invocation. An exhausted target or a checkpoint
        already stopped by early stopping performs no further training.
        """
        self.mode = metrics.monitor_mode(monitor)
        self.monitor = monitor
        m_obj = metrics.TrainingMetrics(monitor=monitor, runtime=utils.runtime_metadata())
        start_dt = datetime.now()
        start_time_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        m_obj.start_time = start_time_str

        dev_gpu = utils.get_device_name(self.device)
        m_obj.device_name = f"{self.device} ({dev_gpu})" if (dev_gpu and dev_gpu.lower() != self.device.type.lower()) else str(self.device)

        t0 = time.time()
        self.should_stop = False
        self._last_spatial_shape = None
        self._variable_shapes_detected = False

        # Consolidate callbacks from list and explicit keyword arguments
        all_callbacks = list(callbacks or [])
        if early_stopping and early_stopping not in all_callbacks:
            all_callbacks.append(early_stopping)
        if checkpoint and checkpoint not in all_callbacks:
            all_callbacks.append(checkpoint)
        if scheduler and scheduler not in all_callbacks:
            all_callbacks.append(scheduler)
        for cb in all_callbacks:
            if isinstance(cb, (EarlyStopping, Checkpoint)):
                cb.monitor = monitor
                cb.mode = self.mode
                cb.best_score = None
                if isinstance(cb, EarlyStopping):
                    cb.bad_epochs = 0
                else:
                    cb.best_saved = False
            if isinstance(cb, torch.optim.lr_scheduler.ReduceLROnPlateau) and cb.mode != self.mode:
                raise ValueError("ReduceLROnPlateau mode must match the monitored metric direction.")

        loaders = (train_loader, val_loader)
        checkpoints = [cb for cb in all_callbacks if isinstance(cb, Checkpoint)]
        stateful_callbacks = [
            cb for cb in all_callbacks
            if not isinstance(cb, Checkpoint)
            and callable(getattr(cb, "state_dict", None))
            and callable(getattr(cb, "load_state_dict", None))
        ]
        callback_types = [
            f"{type(cb).__module__}.{type(cb).__qualname__}"
            for cb in stateful_callbacks
        ]
        history_fields = (
            "train_loss", "val_loss", "val_accuracy",
            "best_epoch", "best_val_loss", "best_score", "stopped_early",
        )

        start_epoch = 0
        if resume_from is not None:
            saved = torch.load(resume_from, map_location="cpu", weights_only=True)
            state = saved.get("training_state")
            if (state is None or state.get("version") != 1
                    or "optimizer_state_dict" not in saved):
                raise ValueError("Checkpoint does not contain supported full training state.")
            if saved["monitor"] != monitor or saved["mode"] != self.mode:
                raise ValueError("Resume requires the same monitored metric.")
            if state["amp_active"] != self.amp_active:
                raise ValueError("Resume requires the same AMP configuration.")
            if "deterministic" in state:
                settings = ("algorithms", "warn_only", "cudnn")
                if self.device.type == "cuda" and state["deterministic"]["algorithms"]:
                    settings += ("cublas_workspace_config",)
                current_settings = utils._deterministic_settings()
                if any(state["deterministic"][key] != current_settings[key] for key in settings):
                    raise ValueError("Resume requires the same deterministic settings.")
            if state["callback_types"] != callback_types:
                raise ValueError("Resume requires matching stateful callback types and order.")
            if len(state["callback_states"]) != len(stateful_callbacks):
                raise ValueError("Checkpoint callback state is incomplete.")

            raw_model = getattr(self.model, "module", self.model)
            raw_model.load_state_dict(saved["model_state_dict"])
            for cb, cb_state in zip(stateful_callbacks, state["callback_states"]):
                cb.load_state_dict(cb_state)
            self.optimizer.load_state_dict(saved["optimizer_state_dict"])
            self.scaler.load_state_dict(state["scaler_state_dict"])
            for field in history_fields:
                setattr(m_obj, field, state["history"][field])
            for cb in checkpoints:
                cb.best_score = m_obj.best_score

            start_epoch = saved["epoch"]
            self.should_stop = m_obj.stopped_early
            _restore_rng_state(state["rng_state"], loaders)
            del saved, state

        try:
            for epoch in range(start_epoch, epochs):
                if self.should_stop:
                    break
                train_loss = self._train_epoch(
                    train_loader,
                    max_batches=max_train_batches,
                    epoch_idx=epoch + 1,
                    total_epochs=epochs,
                    show_progress=progress,
                )
                if monitor == "val_accuracy":
                    self.current_metrics = self.evaluate(val_loader, collect_accuracy=True)
                else:
                    self.current_metrics = {"val_loss": self.evaluate(val_loader)}
                val_loss = self.current_metrics["val_loss"]
                m_obj.train_loss.append(train_loss)
                m_obj.val_loss.append(val_loss)
                if monitor == "val_accuracy":
                    m_obj.val_accuracy.append(self.current_metrics["val_accuracy"])

                score = self.current_metrics[monitor]
                is_best = metrics.is_improvement(score, m_obj.best_score, self.mode)
                if m_obj.best_val_loss is None or val_loss < m_obj.best_val_loss:
                    m_obj.best_val_loss = val_loss
                if is_best:
                    m_obj.best_score = score
                    m_obj.best_epoch = epoch + 1

                should_log = (
                    log_interval <= 1
                    or (epoch + 1) == 1
                    or (epoch + 1) == epochs
                    or (epoch + 1) % log_interval == 0
                    or is_best
                )

                if should_log:
                    accuracy_log = (
                        f" | Validation Accuracy: {self.current_metrics['val_accuracy']:.4f}"
                        if "val_accuracy" in self.current_metrics else ""
                    )
                    best_tag = f" {utils.Colours.BRIGHT_CYAN}★ (Best){utils.Colours.RESET}" if is_best else ""
                    logger.info(
                        f"[BOKeTE] {utils.format_epoch_label(epoch + 1, epochs)} | "
                        f"Train Loss: {utils.Colours.CYAN}{train_loss:.4f}{utils.Colours.RESET} | "
                        f"Val Loss: {utils.Colours.BRIGHT_CYAN}{val_loss:.4f}{utils.Colours.RESET}{accuracy_log}{best_tag}"
                    )

                # Finish state updates before writing the completed-epoch checkpoint.
                for cb in all_callbacks:
                    if isinstance(cb, Checkpoint):
                        continue
                    if hasattr(cb, 'on_epoch_end'):
                        cb.on_epoch_end(self, epoch + 1, val_loss)
                    elif hasattr(cb, 'step'):
                        if isinstance(cb, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            cb.step(score)
                        else:
                            cb.step()

                m_obj.stopped_early = self.should_stop
                if checkpoints:
                    self._checkpoint_state = {
                        "version": 1,
                        "amp_active": self.amp_active,
                        "deterministic": utils._deterministic_settings(),
                        "scaler_state_dict": self.scaler.state_dict(),
                        "callback_types": callback_types,
                        "callback_states": [cb.state_dict() for cb in stateful_callbacks],
                        "history": {
                            field: getattr(m_obj, field) for field in history_fields
                        },
                        "rng_state": _capture_rng_state(loaders),
                    }
                    for cb in checkpoints:
                        cb.on_epoch_end(self, epoch + 1, val_loss)

                if self.should_stop:
                    break

            t1 = time.time()
            end_dt = datetime.now()
            end_time_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
            elapsed = round(t1 - t0, 2)
            m_obj.end_time = end_time_str
            m_obj.duration_seconds = elapsed
            m_obj.runtime = utils.runtime_metadata()

            logger.info("")
            logger.info(
                f"[BOKeTE] Training complete in {utils.format_duration(elapsed)} | "
                f"Start: {start_time_str} | End: {end_time_str}"
            )
        except KeyboardInterrupt:
            t1 = time.time()
            end_dt = datetime.now()
            m_obj.end_time = end_dt.strftime("%Y-%m-%d %H:%M:%S")
            m_obj.duration_seconds = round(t1 - t0, 2)
            logger.warning("")
            logger.warning("[BOKeTE] Training loop interrupted by user (KeyboardInterrupt). Aborting remaining epochs...")
            m_obj.interrupted = True
            m_obj.runtime = utils.runtime_metadata()
            # A signal between list appends must not expose half an epoch.
            completed = min(len(m_obj.train_loss), len(m_obj.val_loss))
            if monitor == "val_accuracy":
                completed = min(completed, len(m_obj.val_accuracy))
            del m_obj.train_loss[completed:]
            del m_obj.val_loss[completed:]
            del m_obj.val_accuracy[completed:]
            selection = metrics.selection_summary(m_obj.as_dict())
            m_obj.best_epoch = selection["best_epoch"]
            m_obj.best_score = selection["best_score"]
            m_obj.best_val_loss = min(m_obj.val_loss) if m_obj.val_loss else None
            raise TrainingInterrupted(m_obj) from None

        return m_obj
