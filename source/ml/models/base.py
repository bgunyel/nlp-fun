from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import torch


class PreTrainedModelPath(Enum):
    bert_tiny = 'prajjwal1/bert-tiny'  # L:2, H:128
    bert_mini = 'prajjwal1/bert-mini'  # L:4, H:256
    bert_small = 'prajjwal1/bert-small'  # L:4, H:512
    bert_medium = 'prajjwal1/bert-medium'  # L:8, H: 512
    bert_base_cased = 'google-bert/bert-base-cased'  # L: 12, H:768
    bert_large_cased = 'google-bert/bert-large-cased'  # L: 24, H: 1024
    electra_small_discriminator = 'google/electra-small-discriminator'
    electra_base_discriminator = 'google/electra-base-discriminator'


@dataclass
class TrainConfig:
    device: torch.device  # This can NOT be None
    module_name: str = None
    backbone: str = None
    dataset_names: list[str] = None
    n_classes: int = None
    n_epochs: int = None
    batch_size: int = None
    mini_batch_size: int = None  # for gradient accumulation in (batch_size // mini_batch_size) steps


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
        self.dataset_names = train_config.dataset_names if train_config.dataset_names is not None else None
        self.n_classes = train_config.n_classes if train_config.n_classes is not None else None
        self.n_epochs = train_config.n_epochs if train_config.n_epochs is not None else None
        self.batch_size = train_config.batch_size if train_config.batch_size is not None else None
        self.mini_batch_size = train_config.mini_batch_size if train_config.mini_batch_size is not None else None
        self.device = train_config.device if train_config.device is not None else None

        self.optimizer_config = optimizer_config

        self.is_data_ready = False
        self.is_model_ready = False
        self.grad_accumulation_steps = self.batch_size // self.mini_batch_size

        self.model = None

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def prepare_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def fit_to_one_batch(self):
        pass

    def get_number_of_model_parameters(self) -> int:
        if not self.is_model_ready:
            raise RuntimeError('Model NOT ready!')

        n_params = sum(
            [p.numel() for p in self.model.parameters(recurse=True)]
        )
        return n_params
