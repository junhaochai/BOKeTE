from dataclasses import asdict
from pathlib import Path
import sys

# Ensure example directory is in sys.path for local imports
example_dir = str(Path(__file__).resolve().parent)
if example_dir not in sys.path:
    sys.path.insert(0, example_dir)

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import bokete
from src.models import SimpleMLP
from src.schema import ExperimentConfig


def main() -> None:
    # 1. Parse CLI arguments & load configuration YAML (forces --config argument)
    cfg = bokete.parse_cli_config(ExperimentConfig)

    # 2. Data Factory Callback
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    ds_kwargs = dict(root=Path("./data"), download=True, transform=transform)

    def loader_factory(c: dict):
        batch_size = c.get("data", {}).get("batch_size", 64)
        val_batch_size = c.get("data", {}).get("val_batch_size", 1000)
        train_data, val_data = random_split(
            datasets.MNIST(train=True, **ds_kwargs),
            [50000, 10000],
            generator=torch.Generator().manual_seed(42),
        )
        t_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
        )
        v_loader = DataLoader(
            val_data,
            batch_size=val_batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            datasets.MNIST(train=False, **ds_kwargs),
            batch_size=val_batch_size,
            shuffle=False,
        )
        return t_loader, v_loader, test_loader

    # 3. Model Factory Callback
    def model_factory(c: dict):
        hidden_dim = c.get("model", {}).get("hidden_dim", 128)
        num_classes = c.get("model", {}).get("num_classes", 10)
        return SimpleMLP(input_dim=784, hidden_dim=hidden_dim, num_classes=num_classes)

    # 4. Test Evaluation Callback
    def test_fn(model: torch.nn.Module, loader: DataLoader, c: dict) -> dict:
        device = next(model.parameters()).device
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.inference_mode():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        acc = round(correct / max(total, 1), 4)
        mean_loss = round(total_loss / max(total, 1), 4)
        return {"accuracy": acc, "test_loss": mean_loss}

    # 5. Create Standardized Trial Runner via bokete.create_trial_runner
    cfg_dict = asdict(cfg)

    runner = bokete.create_trial_runner(
        model_factory=model_factory,
        loader_factory=loader_factory,
        test_fn=test_fn,
    )

    # 6. Execute Experiment Run
    # Primary entry point: automatically runs single-config multi-trial run OR multi-config grid sweep based on YAML
    bokete.run_experiments(
        config=cfg_dict,
        run_fn=runner,
    )


if __name__ == "__main__":
    main()
