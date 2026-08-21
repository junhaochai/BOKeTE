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


def _finalise_plot(fig, ax, save_path: Path, title: str, log_label: str) -> None:
    """Applies BOKeTE plot styling, saves figure to disk, and closes canvas."""
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')

    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"[BOKeTE] Saved {log_label} plot to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save {log_label} plot to {save_path}: {e}")
    finally:
        plt.close(fig)


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
        m = metrics if isinstance(metrics, dict) else metrics.as_dict()
        train_loss = m.get('train_loss', [])
        val_loss = m.get('val_loss', [])
        if best_epoch is None:
            best_epoch = m.get('best_epoch')

    if not train_loss or not val_loss:
        logger.warning("Empty loss lists passed to plot_loss_curves. Skipping plot.")
        return

    # Handle mismatched lengths just in case
    num_epochs = min(len(train_loss), len(val_loss))
    epochs = list(range(1, num_epochs + 1))
    save_path = Path(path)

    # Generate Static Matplotlib Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(epochs, train_loss[:num_epochs], label='Train Loss', color='#2b5c8f', linewidth=2)
    ax.plot(epochs, val_loss[:num_epochs], label='Validation Loss', color='#d95f02', linewidth=2)

    # Draw a line for the best epoch if provided
    if best_epoch is not None and 1 <= best_epoch <= num_epochs:
        best_val = val_loss[best_epoch - 1]
        ax.axvline(x=best_epoch, color='#7570b3', linestyle='--', alpha=0.8, label=f'Best Epoch ({best_epoch})')
        ax.plot(best_epoch, best_val, 'ro', markersize=8, label=f'Best Val Loss ({best_val:.4f})')

    _finalise_plot(fig, ax, save_path, title, log_label="loss curves")


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

    _finalise_plot(fig, ax, save_path, title, log_label="multi-trial loss curves")

