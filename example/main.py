from pathlib import Path
import sys

# Ensure example directory is in sys.path for local imports
example_dir = str(Path(__file__).resolve().parent)
if example_dir not in sys.path:
    sys.path.insert(0, example_dir)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import bokete
from src.models import SimpleMLP
from src.schema import ExperimentConfig



def main() -> None:
    # 1. Parse CLI arguments & load configuration YAML via bokete
    cfg = bokete.parse_cli_config(
        ExperimentConfig,
        default_config=Path(__file__).parent / "configs" / "example.yaml",
    )

    # 2. Setup Environment
    bokete.set_seed(cfg.seed)
    device = bokete.determine_device(verbose=False)

    # 3. Create Timestamped Run Directory
    run_dir = bokete.create_run_directory(cfg)

    # 4. Load MNIST Dataset & DataLoaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    ds_kwargs = dict(root=Path("./data"), download=True, transform=transform)

    train_loader = DataLoader(
        datasets.MNIST(train=True, **ds_kwargs),
        batch_size=cfg.data.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        datasets.MNIST(train=False, **ds_kwargs),
        batch_size=cfg.data.val_batch_size,
        shuffle=False,
    )

    # 5. Model, Loss, Optimizer
    model = SimpleMLP(
        input_dim=784,
        hidden_dim=cfg.model.hidden_dim,
        num_classes=cfg.model.num_classes,
    ).to(device)
    
    criterion = getattr(nn, cfg.training.loss)()
    optimizer = getattr(torch.optim, cfg.training.optimizer)(model.parameters(), lr=cfg.training.lr)

    # 6. Train Model via Trainer Wrapper
    trainer = bokete.Trainer(model, criterion, optimizer, device=device)
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.training.epochs,
        early_stopping=bokete.EarlyStopping(patience=cfg.training.patience),
        checkpoint=bokete.Checkpoint(run_dir / "checkpoints"),
    )

    # 7. Generate & Save Markdown Report + Loss Graph PNG/HTML
    bokete.experiment_report(
        config=cfg,
        metrics=history,
        save_path=run_dir,
    )


if __name__ == "__main__":
    main()
