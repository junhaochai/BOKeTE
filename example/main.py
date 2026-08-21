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
    # 1. Load Configuration YAML into typed ExperimentConfig via bokete
    config_path = Path(__file__).parent / "configs" / "example.yaml"
    cfg = bokete.load_config_as(config_path, ExperimentConfig)

    # 2. Setup Environment
    bokete.set_seed(cfg.seed)
    device = bokete.determine_device(verbose=False)

    # 3. Create Timestamped Run Directory
    run_dir = Path(
        bokete.create_run_directory(
            base_dir=Path(__file__).parent / "results",
            experiment_name=cfg.experiment_name,
        )
    )

    # 4. Load MNIST Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    data_dir = Path("./data")
    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    val_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.data.val_batch_size, shuffle=False
    )

    # 5. Model, Loss, Optimizer
    model = SimpleMLP(
        input_dim=784,
        hidden_dim=cfg.model.hidden_dim,
        num_classes=cfg.model.num_classes,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    callbacks = [
        bokete.EarlyStopping(patience=cfg.training.patience),
        bokete.Checkpoint(directory=run_dir / "checkpoints"),
    ]

    # 6. Train Model via Trainer Wrapper
    trainer = bokete.Trainer(model, criterion, optimizer, device=device)
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.training.epochs,
        callbacks=callbacks,
        progress=True,
    )

    # 7. Generate & Save Markdown Report + Loss Graph PNG/HTML
    report_path = run_dir / "report.md"
    bokete.experiment_report(
        config=cfg,
        metrics=history,
        save_path=str(report_path),
        title="MNIST MLP Training Run Demo",
    )

    print(f"[BOKeTE] Run complete! Saved report to: {report_path}")


if __name__ == "__main__":
    main()
