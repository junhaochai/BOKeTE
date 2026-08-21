"""
Core training machinery for bokete.

Key Classes & Functions:
  - Trainer      : Main training loop engine handling fit(), epochs, AMP, and callback hooks.
  - EarlyStopping: Callback for halting training when validation loss stops improving.
  - Checkpoint   : Callback for saving best.pt and last.pt model state dictionaries.
  - evaluate     : Computes evaluation loss over a DataLoader without tracking gradients.
"""

import os
import random
from pathlib import Path
from typing import Callable, Optional, Any, List

import logging
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

import time
from datetime import datetime

from bokete.metrics import TrainingMetrics
from bokete.utils import set_seed, determine_device, log_model_info, get_device_name, format_duration

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Signals that training should stop once validation loss has not improved
    by at least min_delta for `patience` consecutive epochs.
    """

    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.bad_epochs = 0

    def step(self, val_loss):
        """Record this epoch's validation loss. Returns True when patience is exhausted."""
        if self.patience is None or self.patience <= 0:
            return False
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def on_epoch_end(self, trainer: Any, epoch: int, val_loss: float) -> None:
        """Callback hook for Trainer fit loop."""
        if self.step(val_loss):
            trainer.should_stop = True


class Checkpoint:
    """
    Saves model weights during training: 'last.pt' every epoch and 'best.pt'
    whenever validation loss improves. Files contain a dict with the epoch,
    validation loss and model state_dict.
    """

    def __init__(self, directory, save_best=True, save_last=True, enabled=True):
        self.directory = Path(directory)
        self.save_best = save_best
        self.save_last = save_last
        self.enabled = enabled
        self.best_loss = float('inf')

    def update(self, model, epoch, val_loss, optimizer=None):
        if not self.enabled or not (self.save_best or self.save_last):
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        # Extract inner model from multi-GPU wrapper (DataParallel/DDP), or fallback to raw model on single-GPU/CPU
        raw_model = getattr(model, 'module', model)
        state = {
            'epoch': epoch,
            'val_loss': val_loss,
            'model_state_dict': raw_model.state_dict(),
        }
        if optimizer is not None:
            state['optimizer_state_dict'] = optimizer.state_dict()

        if self.save_last:
            torch.save(state, self.directory / 'last.pt')
        if self.save_best and val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(state, self.directory / 'best.pt')

    def on_epoch_end(self, trainer: Any, epoch: int, val_loss: float) -> None:
        """Callback hook for Trainer fit loop."""
        self.update(trainer.model, epoch, val_loss, optimizer=trainer.optimizer)


def _default_prepare_batch(batch, device):
    """Default batch handling: assumes (inputs, targets) pairs."""
    inputs, targets = batch
    return inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)


def evaluate(model, loader, criterion, device=None, prepare_batch=None, amp_active=False):
    """Compute the mean loss of `model` over `loader` without tracking gradients."""
    device = torch.device(device) if device is not None else determine_device()

    if prepare_batch is None:
        prepare_batch = _default_prepare_batch

    model.eval()
    running_loss = 0.0
    batches = 0

    with torch.inference_mode():
        for batch in loader:
            inputs, targets = prepare_batch(batch, device)
            with torch.autocast(device_type=device.type, enabled=amp_active):
                outputs = model(inputs)
                running_loss += criterion(outputs, targets).item()
            batches += 1

    return running_loss / max(batches, 1)


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
        self.device = torch.device(device) if device is not None else determine_device()
        self.prepare_batch = prepare_batch or _default_prepare_batch
        self.max_grad_norm = max_grad_norm

        self.model.to(self.device)
        log_model_info(self.model, log_layers=log_model_structure)

        # AMP only applies on CUDA; on CPU both autocast and the scaler become no-ops.
        self.amp_active = amp and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp_active)

        # Track spatial shape consistency across batches for safe cuDNN benchmarking
        self._last_spatial_shape = None
        self._variable_shapes_detected = False

        self.should_stop = False

    def evaluate(self, loader: DataLoader) -> float:
        """Compute the mean loss of `model` over `loader` using Trainer configuration."""
        return evaluate(
            self.model,
            loader,
            self.criterion,
            device=self.device,
            prepare_batch=self.prepare_batch,
            amp_active=self.amp_active,
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
        running_loss = 0.0
        batches_run = 0

        desc = f" Epoch {epoch_idx}/{total_epochs}" if (epoch_idx and total_epochs) else " Training"
        batch_pbar = tqdm.tqdm(
            train_loader,
            desc=desc,
            dynamic_ncols=True,
            leave=False,
            disable=not show_progress,
        )

        for batch in batch_pbar:
            if max_batches is not None and batches_run >= max_batches:
                break

            inputs, targets = self.prepare_batch(batch, self.device)

            # Auto-detect spatial shape consistency for cuDNN benchmarking
            if self.device.type == 'cuda' and not self._variable_shapes_detected:
                spatial_shape = tuple(inputs.shape[1:])
                if self._last_spatial_shape is None:
                    self._last_spatial_shape = spatial_shape
                    torch.backends.cudnn.benchmark = True
                elif spatial_shape != self._last_spatial_shape:
                    self._variable_shapes_detected = True
                    torch.backends.cudnn.benchmark = False

            self.optimizer.zero_grad()

            with torch.autocast(device_type=self.device.type, enabled=self.amp_active):
                outputs = self.model(inputs)
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
            running_loss += loss_val
            batches_run += 1
            avg_loss = running_loss / batches_run
            batch_pbar.set_postfix(loss=f"{avg_loss:.4f}")

        return round(running_loss / max(batches_run, 1), 4)

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
    ) -> TrainingMetrics:
        """Run the training/validation loop over the specified number of epochs.

        Args:
            train_loader (DataLoader): DataLoader for the training phase.
            val_loader (DataLoader): DataLoader for the validation phase.
            epochs (int): Maximum number of epochs to run.
            callbacks (list, optional): List of callback objects (e.g. EarlyStopping, Checkpoint, Schedulers).
            scheduler (optional): Learning rate scheduler; `.step()` is called once per epoch.
            early_stopping (EarlyStopping, optional): EarlyStopping instance.
            checkpoint (Checkpoint, optional): Checkpoint instance.
            progress (bool): If True, shows a progress bar.
            max_train_batches (int, optional): Cap on training batches per epoch.

        Returns:
            TrainingMetrics: An object containing lists of training and validation losses.
        """
        metrics = TrainingMetrics()
        start_dt = datetime.now()
        start_time_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        metrics.start_time = start_time_str

        dev_gpu = get_device_name(self.device)
        metrics.device_name = f"{self.device} ({dev_gpu})" if (dev_gpu and dev_gpu.lower() != self.device.type.lower()) else str(self.device)

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

        try:
            with tqdm.tqdm(range(epochs), desc=" Overall Progress", dynamic_ncols=True, leave=True, disable=not progress) as pbar:
                for epoch in pbar:
                    train_loss = self._train_epoch(
                        train_loader,
                        max_batches=max_train_batches,
                        epoch_idx=epoch + 1,
                        total_epochs=epochs,
                        show_progress=progress,
                    )
                    metrics.train_loss.append(train_loss)

                    val_loss = round(self.evaluate(val_loader), 4)
                    metrics.val_loss.append(val_loss)
                    pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

                    if metrics.best_val_loss is None or val_loss < metrics.best_val_loss:
                        metrics.best_val_loss = val_loss
                        metrics.best_epoch = epoch + 1

                    # Process callbacks at end of epoch
                    for cb in all_callbacks:
                        if hasattr(cb, 'on_epoch_end'):
                            cb.on_epoch_end(self, epoch + 1, val_loss)
                        elif hasattr(cb, 'step'):
                            if isinstance(cb, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                cb.step(val_loss)
                            else:
                                cb.step()

                    if self.should_stop:
                        metrics.stopped_early = True
                        break

            t1 = time.time()
            end_dt = datetime.now()
            end_time_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
            elapsed = round(t1 - t0, 2)
            metrics.end_time = end_time_str
            metrics.duration_seconds = elapsed

            logger.info(
                f"[BOKeTE] Training complete in {format_duration(elapsed)} | "
                f"Start: {start_time_str} | End: {end_time_str}"
            )
        except KeyboardInterrupt:
            t1 = time.time()
            end_dt = datetime.now()
            metrics.end_time = end_dt.strftime("%Y-%m-%d %H:%M:%S")
            metrics.duration_seconds = round(t1 - t0, 2)
            logger.warning("\n[BOKeTE] Training loop interrupted by user (KeyboardInterrupt). Aborting remaining epochs...")
            metrics.interrupted = True
            raise

        return metrics
