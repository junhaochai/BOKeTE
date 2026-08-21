"""
Core training machinery: the train() loop, evaluation, early stopping,
checkpointing, and seeding/device utilities
"""

import os
import random
from pathlib import Path
from typing import Callable, Optional, Any

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from bokete.metrics import TrainingMetrics


def set_seed(seed: int, deterministic: bool = False):
    """Seed Python, NumPy and PyTorch (CPU and all CUDA devices) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Enforce deterministic behaviour for convolutions and matrix multiplications
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def determine_device():
    """Return the best available PyTorch device: CUDA, MPS (Mac), or CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


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
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


class Checkpoint:
    """
    Saves model weights during training: 'last.pt' every epoch and 'best.pt'
    whenever validation loss improves. Files contain a dict with the epoch,
    validation loss and model state_dict.
    """

    def __init__(self, directory, save_best=True, save_last=True):
        self.directory = Path(directory)
        self.save_best = save_best
        self.save_last = save_last
        self.best_loss = float('inf')

    def update(self, model, epoch, val_loss, optimizer=None):
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
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device(device) if device is not None else determine_device()
        self.prepare_batch = prepare_batch or _default_prepare_batch
        self.max_grad_norm = max_grad_norm

        self.model.to(self.device)

        # AMP only applies on CUDA; on CPU both autocast and the scaler become no-ops.
        self.amp_active = amp and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp_active)

        # Track spatial shape consistency across batches for safe cuDNN benchmarking
        self._last_spatial_shape = None
        self._variable_shapes_detected = False

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        *,
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
            scheduler (optional): Learning rate scheduler; `.step()` is called once per epoch.
            early_stopping (EarlyStopping, optional): EarlyStopping instance.
            checkpoint (Checkpoint, optional): Checkpoint instance.
            progress (bool): If True, shows a progress bar.
            max_train_batches (int, optional): Cap on training batches per epoch.

        Returns:
            TrainingMetrics: An object containing lists of training and validation losses.
        """
        metrics = TrainingMetrics()
        self._last_spatial_shape = None
        self._variable_shapes_detected = False

        with tqdm.tqdm(range(epochs), desc=" Epochs", dynamic_ncols=True, leave=False, disable=not progress) as pbar:
            for epoch in pbar:
                self.model.train()
                running_loss = 0.0
                batches_run = 0

                for batch in train_loader:
                    if max_train_batches is not None and batches_run >= max_train_batches:
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

                    running_loss += loss.item()
                    batches_run += 1

                # Mean over batches actually run, not len(train_loader)
                train_loss = round(running_loss / max(batches_run, 1), 4)
                metrics.train_loss.append(train_loss)

                val_loss = round(evaluate(self.model, val_loader, self.criterion,
                                          device=self.device, prepare_batch=self.prepare_batch,
                                          amp_active=self.amp_active), 4)
                metrics.val_loss.append(val_loss)
                pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

                if metrics.best_val_loss is None or val_loss < metrics.best_val_loss:
                    metrics.best_val_loss = val_loss
                    metrics.best_epoch = epoch + 1

                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        if val_loss is not None:
                            scheduler.step(val_loss)
                    else:
                        scheduler.step()

                if checkpoint is not None:
                    checkpoint.update(self.model, epoch + 1, val_loss, optimizer=self.optimizer)

                if early_stopping is not None and early_stopping.step(val_loss):
                    metrics.stopped_early = True
                    break

        return metrics
