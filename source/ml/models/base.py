from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch


@dataclass
class TrainConfig:
    device: torch.device  # This can NOT be None
    module_name: str = None
    backbone: str = None
    max_length: int = None  # max number of tokens for the backbone model
    dataset_names: list[str] = None
    n_epochs: int = None
    batch_size: int = None



@dataclass
class OptimizerConfig:
    lr: float = None
    weight_decay: float = None
    betas: tuple[float, float] = None
    eps: float = None


class TrainerBase(ABC):

    def __init__(self, train_config: TrainConfig, optimizer_config: OptimizerConfig):
        self.module_name = train_config.module_name if train_config.module_name is not None else None
        self.backbone = train_config.backbone if train_config.backbone is not None else None
        self.max_length = train_config.max_length if train_config.max_length is not None else None
        self.dataset_names = train_config.dataset_names if train_config.dataset_names is not None else None
        self.n_epochs = train_config.n_epochs if train_config.n_epochs is not None else None
        self.batch_size = train_config.batch_size if train_config.batch_size is not None else None
        self.device = train_config.device if train_config.device is not None else None

        self.optimizer_config = optimizer_config

        self.is_data_ready = False
        self.is_model_ready = False

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def prepare_model(self):
        pass

    @abstractmethod
    def train(self):
        pass
