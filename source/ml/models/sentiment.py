from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from source.ml.models.base import TrainerBase, TrainConfig
from source.ml.data import *


@dataclass
class ModelConfig:
    embeddings_dim: int = 64  # length of embedding vectors
    n_hidden_units: int = 128

@dataclass
class OptimizerConfig:
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8


class TheModel(nn.Module):
    def __init__(self, config: ModelConfig, vocabulary_size: int, num_classes: int):
        super().__init__()
        self.name = 'Sentiment Analysis'


class TheTrainer(TrainerBase):
    def __init__(self, train_config: TrainConfig):
        super().__init__(train_config=train_config)
        self.name = 'Sentiment Analysis'

        self.tokenizer = AutoTokenizer.from_pretrained(train_config.backbone)

        self.train_data_1 = DynaSent(data_round=1, data_split='train', tokenizer=self.tokenizer, max_length=train_config.max_length)
        self.valid_data_1 = DynaSent(data_round=1, data_split='validation', tokenizer=self.tokenizer, max_length=train_config.max_length)
        self.train_data_2 = DynaSent(data_round=2, data_split='train', tokenizer=self.tokenizer, max_length=train_config.max_length)
        self.valid_data_2 = DynaSent(data_round=2, data_split='validation', tokenizer=self.tokenizer, max_length=train_config.max_length)
        dummy = -32

    def train(self):
        pass


