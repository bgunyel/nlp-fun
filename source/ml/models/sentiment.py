from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from source.ml.models.base import TrainerBase, TrainConfig, OptimizerConfig
from source.ml.datasets import *


@dataclass
class ModelConfig:
    backbone: str
    n_hidden_units: int


class TheModel(nn.Module):
    def __init__(self, config: ModelConfig, num_classes: int):
        super().__init__()
        self.name = 'Sentiment Analysis'
        self.backbone = AutoModel.from_pretrained(config.backbone)
        self.embedding_dim = self.backbone.embeddings.word_embeddings.embedding_dim

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        dummy = -32


class TheTrainer(TrainerBase):
    def __init__(self, train_config: TrainConfig, optimizer_config: OptimizerConfig):
        super().__init__(train_config=train_config, optimizer_config=optimizer_config)
        self.name = 'Sentiment Analysis'
        self.model, self.tokenizer = self.prepare_model()
        self.train_data, self.valid_data = self.prepare_data()

    def prepare_data(self):
        if not self.is_model_ready:
            raise RuntimeError('Tokenizer must be prepared before training!')

        self.is_data_ready = True
        train_data = DynaSent(data_round=1, data_split='train', tokenizer=self.tokenizer, max_length=self.max_length)
        valid_data = DynaSent(data_round=1, data_split='validation', tokenizer=self.tokenizer, max_length=self.max_length)

        return train_data, valid_data

    def prepare_model(self) -> tuple[nn.Module, nn.Module]:
        model = TheModel(config=ModelConfig(backbone=self.backbone, n_hidden_units=128), num_classes=0)
        tokenizer = AutoTokenizer.from_pretrained(self.backbone)
        self.is_model_ready = True
        return model, tokenizer

    def train(self):
        if (not self.is_data_ready) or (not self.is_model_ready):
            raise RuntimeError('Data and Model must be ready before training!')

        train_loader = DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )

        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.optimizer_config.lr,
                                      weight_decay=self.optimizer_config.weight_decay,
                                      betas=self.optimizer_config.betas,
                                      eps=self.optimizer_config.eps)
        self.model.train()
        epoch_iters = len(train_loader)
        eval_iters = int(epoch_iters * 0.1)

        for epoch in range(self.n_epochs):
            tqdm_train_loader = tqdm(train_loader, unit="batch", desc=f'Epoch: {epoch}')

            for iteration, data_dict in enumerate(tqdm_train_loader):
                input_ids = data_dict['input_ids'].to(self.device)
                attention_mask = data_dict['attention_mask'].to(self.device)
                label = data_dict['label'].to(self.device)



