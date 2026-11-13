import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron for 28x28 image classification."""

    def __init__(
        self, input_dim: int = 784, hidden_dim: int = 128, num_classes: int = 10
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
