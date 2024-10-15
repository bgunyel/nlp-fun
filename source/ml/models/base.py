import importlib
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
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
    name: str = None
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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Modified from https://github.com/karpathy/nanoGPT/blob/9755682b981a45507f6eb9b11eadef8cb83cebd5/model.py#L263
        """

        if not self.is_model_ready:
            raise RuntimeError('Model must be prepared before Optimizer!')

        # start with all the candidate parameters
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layer norms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.optimizer_config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)

        print('Trainer Optimizer Config:')
        print(f"Number of decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"Number of non-decayed parameter tensors: {len(no_decay_params)}, with {num_no_decay_params:,} parameters")

        module = importlib.import_module(name='torch.optim')
        class_ = getattr(module, self.optimizer_config.name)

        # Use the fused version of the optimizer if it is available
        fused_available = 'fused' in inspect.signature(class_).parameters
        use_fused = fused_available and self.device.type == 'cuda'
        optimizer_args = asdict(self.optimizer_config)
        optimizer_args['fused'] = use_fused
        optimizer_args.pop('name')

        optimizer = class_(params=optim_groups, **optimizer_args)

        return optimizer
