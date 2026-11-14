from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DataConfig:
    batch_size: int = 64
    val_batch_size: int = 1000


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    num_classes: int = 10


@dataclass
class TrainingConfig:
    epochs: int = 5
    lr: float = 1e-3
    optimizer: str = "Adam"
    loss: str = "CrossEntropyLoss"
    amp: bool = False
    max_grad_norm: Optional[float] = None
    log_interval: int = 1
    patience: Optional[int] = 3
    save_checkpoints: bool = True
    monitor: str = "val_accuracy"
    resume_from: Dict[str, str] = field(default_factory=dict)
    resume_best_from: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    experiment_name: str = "mnist_mlp_demo"
    seed: int = 42
    deterministic: bool = False
    device: Optional[str] = None
    trials: int = 3
    param_grid: Optional[Dict[str, List[Any]]] = None
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
