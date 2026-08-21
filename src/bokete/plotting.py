"""
Visualization utilities for training and validation metrics.

Key Functions:
  - plot_loss_curves: Plots per-epoch training/validation loss to a static PNG.
"""

from pathlib import Path
import logging
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_loss_curves(
    train_loss=None,
    val_loss=None,
    path="graph.png",
    title='Training and Validation Loss',
    best_epoch=None,
    metrics=None,
):
    """
    Plot per-epoch training and validation loss and save a static PNG to `path`, 
    along with a companion interactive HTML file.

    Accepts either individual `train_loss` / `val_loss` lists or a `TrainingMetrics` / dict instance.
    """
    # Support passing TrainingMetrics or dict as first positional argument
    if hasattr(train_loss, 'train_loss') or (isinstance(train_loss, dict) and 'train_loss' in train_loss):
        metrics = train_loss
        if isinstance(val_loss, (str, Path)):
            path = val_loss
        train_loss = None
        val_loss = None

    if metrics is not None:
        if hasattr(metrics, 'train_loss'):
            train_loss = metrics.train_loss
            val_loss = metrics.val_loss
            if best_epoch is None:
                best_epoch = getattr(metrics, 'best_epoch', None)
        elif isinstance(metrics, dict):
            train_loss = metrics.get('train_loss', [])
            val_loss = metrics.get('val_loss', [])
            if best_epoch is None:
                best_epoch = metrics.get('best_epoch')

    if not train_loss or not val_loss:
        logger.warning("Empty loss lists passed to plot_loss_curves. Skipping plot.")
        return

    # Handle mismatched lengths just in case
    num_epochs = min(len(train_loss), len(val_loss))
    epochs = list(range(1, num_epochs + 1))
    save_path = Path(path)

    # 1. Generate Static Matplotlib Plot
    try:
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(epochs, train_loss[:num_epochs], label='Train Loss', color='#2b5c8f', linewidth=2)
        ax.plot(epochs, val_loss[:num_epochs], label='Validation Loss', color='#d95f02', linewidth=2)

        # Draw a line for the best epoch if provided
        if best_epoch is not None and 1 <= best_epoch <= num_epochs:
            best_val = val_loss[best_epoch - 1]
            ax.axvline(x=best_epoch, color='#7570b3', linestyle='--', alpha=0.8, label=f'Best Epoch ({best_epoch})')
            ax.plot(best_epoch, best_val, 'ro', markersize=8, label=f'Best Val Loss ({best_val:.4f})')

        ax.set_title(title, fontsize=14, pad=15)
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right')

        # Ensure the destination directory exists before saving
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"[BOKeTE] Saved loss curves plot to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save static loss curve plot to {path}: {e}")
    finally:
        if 'fig' in locals():
            plt.close(fig)


def plot_multi_trial_loss_curves(
    all_trial_metrics: list,
    path="multi_trial_loss.png",
    title="Multi-Trial Loss Curves (Mean ± Std)",
):
    """
    Plots multi-trial training and validation loss curves with mean and std shaded bands.
    """
    train_losses = [m['train_loss'] for m in all_trial_metrics if isinstance(m, dict) and 'train_loss' in m and m['train_loss']]
    val_losses = [m['val_loss'] for m in all_trial_metrics if isinstance(m, dict) and 'val_loss' in m and m['val_loss']]

    if not train_losses or not val_losses:
        logger.warning("Empty multi-trial loss lists passed to plot_multi_trial_loss_curves. Skipping plot.")
        return

    min_epochs = min(min(len(t) for t in train_losses), min(len(v) for v in val_losses))
    epochs = np.arange(1, min_epochs + 1)

    t_arr = np.array([t[:min_epochs] for t in train_losses])
    v_arr = np.array([v[:min_epochs] for v in val_losses])

    t_mean, t_std = np.mean(t_arr, axis=0), np.std(t_arr, axis=0)
    v_mean, v_std = np.mean(v_arr, axis=0), np.std(v_arr, axis=0)

    save_path = Path(path)

    try:
        fig, ax = plt.subplots(figsize=(10, 5))

        # Individual faint trial lines
        for i in range(len(t_arr)):
            ax.plot(epochs, t_arr[i], color='#2b5c8f', alpha=0.2, linewidth=1)
            ax.plot(epochs, v_arr[i], color='#d95f02', alpha=0.2, linewidth=1)

        # Mean lines
        ax.plot(epochs, t_mean, label='Mean Train Loss', color='#2b5c8f', linewidth=2.5)
        ax.plot(epochs, v_mean, label='Mean Validation Loss', color='#d95f02', linewidth=2.5)

        # Shaded ±1 Std bands
        ax.fill_between(epochs, t_mean - t_std, t_mean + t_std, color='#2b5c8f', alpha=0.15)
        ax.fill_between(epochs, v_mean - v_std, v_mean + v_std, color='#d95f02', alpha=0.15)

        ax.set_title(title, fontsize=14, pad=15)
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right')

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"[BOKeTE] Saved multi-trial loss curves plot to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save multi-trial loss curve plot to {path}: {e}")
    finally:
        if 'fig' in locals():
            plt.close(fig)

