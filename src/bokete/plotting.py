import json
from pathlib import Path
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_loss_curves(train_loss, val_loss, path, title='Training and Validation Loss', best_epoch=None):
    """
    Plot per-epoch training and validation loss and save a static PNG to `path`, 
    along with a companion interactive HTML file.
    """
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
    except Exception as e:
        logger.error(f"Failed to save static loss curve plot to {path}: {e}")
    finally:
        if 'fig' in locals():
            plt.close(fig)

    # 2. Generate Interactive Chart.js HTML Plot
    try:
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>__TITLE__</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(17, 24, 39, 0.75);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --accent-primary: #3b82f6;
            --accent-secondary: #f97316;
            --btn-bg: rgba(59, 130, 246, 0.15);
            --btn-hover: rgba(59, 130, 246, 0.3);
        }

        body {
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.12) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(249, 115, 22, 0.08) 0px, transparent 50%);
            color: var(--text-primary);
            font-family: 'Outfit', sans-serif;
            min-height: 100vh;
            margin: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            box-sizing: border-box;
        }

        .card {
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            padding: 30px;
            width: 100%;
            max-width: 900px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 15px;
        }

        h2 {
            margin: 0;
            font-weight: 800;
            font-size: 1.6rem;
            background: linear-gradient(to right, #3b82f6, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .btn {
            background: var(--btn-bg);
            border: 1px solid var(--accent-primary);
            color: var(--text-primary);
            padding: 10px 20px;
            border-radius: 10px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            font-family: 'Outfit', sans-serif;
            font-size: 0.9rem;
            outline: none;
        }

        .btn:hover {
            background: var(--btn-hover);
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(59, 130, 246, 0.25);
        }

        .btn:active {
            transform: translateY(0);
        }

        .canvas-container {
            position: relative;
            width: 100%;
            height: 480px;
        }
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <h2>__TITLE__</h2>
            <button class="btn" onclick="downloadPNG()">Export as PNG</button>
        </div>
        <div class="canvas-container">
            <canvas id="lossChart"></canvas>
        </div>
    </div>
    <script>
        const ctx = document.getElementById('lossChart').getContext('2d');
        const bestEpochVal = __BEST_EPOCH__;

        const bestEpochPlugin = {
            id: 'bestEpochLine',
            beforeDraw: (chart) => {
                if (!bestEpochVal) return;
                
                const ctx = chart.ctx;
                const xAxis = chart.scales.x;
                const yAxis = chart.scales.y;
                const xPixel = xAxis.getPixelForValue(bestEpochVal);
                
                if (xPixel === undefined || isNaN(xPixel)) return;
                
                ctx.save();
                ctx.beginPath();
                ctx.moveTo(xPixel, yAxis.top);
                ctx.lineTo(xPixel, yAxis.bottom);
                ctx.lineWidth = 1.5;
                ctx.strokeStyle = 'rgba(117, 112, 179, 0.8)';
                ctx.setLineDash([6, 6]);
                ctx.stroke();
                ctx.restore();
            }
        };

        const chart = new Chart(ctx, {
            type: 'line',
            plugins: [bestEpochPlugin],
            data: {
                labels: __EPOCHS__,
                datasets: [
                    {
                        label: 'Train Loss',
                        data: __TRAIN_LOSS__,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.03)',
                        borderWidth: 2.5,
                        pointRadius: 2.5,
                        pointHoverRadius: 5.5,
                        tension: 0.1,
                        fill: true
                    },
                    {
                        label: 'Validation Loss',
                        data: __VAL_LOSS__,
                        borderColor: '#f97316',
                        backgroundColor: 'rgba(249, 115, 22, 0.03)',
                        borderWidth: 2.5,
                        pointRadius: 2.5,
                        pointHoverRadius: 5.5,
                        tension: 0.1,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#9ca3af',
                            font: { family: 'Outfit', size: 12, weight: '600' }
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: '#1f2937',
                        titleColor: '#f3f4f6',
                        bodyColor: '#9ca3af',
                        borderColor: 'rgba(255, 255, 255, 0.08)',
                        borderWidth: 1,
                        titleFont: { family: 'Outfit', weight: 'bold', size: 13 },
                        bodyFont: { family: 'Outfit', size: 12 },
                        padding: 12,
                        cornerRadius: 8
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.04)' },
                        ticks: { color: '#9ca3af', font: { family: 'Outfit' } },
                        title: { display: true, text: 'Epochs', color: '#9ca3af', font: { family: 'Outfit', size: 12, weight: '600' } }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.04)' },
                        ticks: { color: '#9ca3af', font: { family: 'Outfit' } },
                        title: { display: true, text: 'Loss', color: '#9ca3af', font: { family: 'Outfit', size: 12, weight: '600' } }
                    }
                }
            }
        });

        function downloadPNG() {
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = chart.width;
            tempCanvas.height = chart.height;
            const tempCtx = tempCanvas.getContext('2d');

            tempCtx.fillStyle = '#0b0f19';
            tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
            tempCtx.drawImage(chart.canvas, 0, 0);

            const link = document.createElement('a');
            link.download = 'loss_curves.png';
            link.href = tempCanvas.toDataURL('image/png');
            link.click();
        }
    </script>
</body>
</html>"""

        html_content = (html_template
                        .replace('__TITLE__', title)
                        .replace('__EPOCHS__', json.dumps(epochs))
                        .replace('__TRAIN_LOSS__', json.dumps(train_loss[:num_epochs]))
                        .replace('__VAL_LOSS__', json.dumps(val_loss[:num_epochs]))
                        .replace('__BEST_EPOCH__', str(best_epoch) if (best_epoch is not None and 1 <= best_epoch <= num_epochs) else 'null'))

        html_path = save_path.with_suffix('.html')
        html_path.write_text(html_content, encoding='utf-8')
    except Exception as e:
        logger.error(f"Failed to save interactive loss curve plot to {path.with_suffix('.html') if isinstance(path, Path) else path}: {e}")
