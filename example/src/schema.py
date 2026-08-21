from dataclasses import dataclass, field


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
    patience: int = 3


@dataclass
class ExperimentConfig:
    experiment_name: str = "mnist_mlp_demo"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
