from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TrainConfig:
    model_name: str = None
    backbone: str = None
    max_length: int = None  # max number of tokens for the backbone model
    dataset_names: list[str] = None
    n_epochs: int = None
    batch_size: int = None


class TrainerBase(ABC):

    def __init__(self, train_config: TrainConfig):
        self.model_name = train_config.model_name if train_config.model_name is not None else None
        self.backbone = train_config.backbone if train_config.backbone is not None else None
        self.dataset_names = train_config.dataset_names if train_config.dataset_names is not None else None
        self.n_epochs = train_config.n_epochs if train_config.n_epochs is not None else None
        self.batch_size = train_config.batch_size if train_config.batch_size is not None else None

    @abstractmethod
    def train(self):
        pass
