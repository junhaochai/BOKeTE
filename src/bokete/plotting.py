"""
Visualization utilities for training and validation metrics.

Key Functions:
  - plot_loss_curves: Plots per-epoch training/validation loss to a static PNG.
"""

from pathlib import Path
import logging
import matplotlib.pyplot as plt

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
