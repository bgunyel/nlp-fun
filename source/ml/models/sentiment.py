from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm

from source.ml.models.base import TrainerBase, TrainConfig, OptimizerConfig
from source.ml.datasets import *


@dataclass
class ModelConfig:
    backbone: str
    fc_hidden_size: int = 1024
    dropout_prob: float = 0.25


class TheModel(nn.Module):
    def __init__(self, config: ModelConfig, num_classes: int):
        super().__init__()
        self.name = 'Sentiment Analysis'
        self.backbone = AutoModel.from_pretrained(config.backbone)
        self.backbone_hidden_size = self.backbone.config.hidden_size
        self.fc_hidden_size = config.fc_hidden_size
        self.dropout_prob = config.dropout_prob

        self.dense = nn.Linear(self.backbone_hidden_size, self.fc_hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(self.dropout_prob)
        self.out_fc = nn.Linear(self.fc_hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        backbone_out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        x = backbone_out.last_hidden_state[:, 0, :]

        out = self.dropout(x)
        out = self.dense(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.out_fc(out)
        return out  # Logits


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
        train_data = DynaSent(data_round=1, data_split='train', tokenizer=self.tokenizer)
        valid_data = DynaSent(data_round=1, data_split='validation', tokenizer=self.tokenizer)

        return train_data, valid_data

    def prepare_model(self) -> tuple[nn.Module, nn.Module]:
        model = TheModel(config=ModelConfig(backbone=self.backbone), num_classes=self.n_classes).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.backbone)
        self.is_model_ready = True
        return model, tokenizer

    def train(self):
        if (not self.is_data_ready) or (not self.is_model_ready):
            raise RuntimeError('Data and Model must be ready before training!')

        train_loader = DataLoader(
            dataset=self.train_data,
            batch_size=self.mini_batch_size,
            shuffle=True,
            num_workers=1,   # TODO: 4
            pin_memory=True,
            drop_last=True
        )

        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.optimizer_config.lr,
                                      weight_decay=self.optimizer_config.weight_decay,
                                      betas=self.optimizer_config.betas,
                                      eps=self.optimizer_config.eps)
        self.model.train()
        epoch_iters = len(train_loader)
        eval_iters = int(epoch_iters * 0.1)

        iteration = 0 # Keeps track of mini-batches
        step = 0 # Keeps track of batches

        for epoch in range(self.n_epochs):
            # tqdm_train_loader = tqdm(train_loader, unit="batch", desc=f'Epoch: {epoch}')
            optimizer.zero_grad()
            loss_accumulated = 0

            for data_dict in train_loader:
                input_ids = data_dict['input_ids'].to(self.device)
                attention_mask = data_dict['attention_mask'].to(self.device)
                label = data_dict['label'].to(self.device)
                idx = data_dict['idx']

                with torch.autocast(device_type=self.device.type):
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = F.cross_entropy(input=logits, target=label, reduction='mean')
                loss = loss / self.grad_accumulation_steps
                loss_accumulated += loss.detach()
                loss.backward()

                if (iteration + 1) % self.grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    print(f'\t\tStep: {step} | Loss: {loss_accumulated.item()}')
                    loss_accumulated = 0
                    step += 1

                iteration += 1


