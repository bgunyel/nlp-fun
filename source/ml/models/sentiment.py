from dataclasses import dataclass
from math import floor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm

from source.ml.models.base import TrainerBase, TrainConfig, OptimizerConfig
from source.ml.datasets import *
from source.ml.utils import evaluate, get_lr


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
        tokenizer.model_max_length = model.backbone.embeddings.position_embeddings.num_embeddings  # to be on the safe side
        self.is_model_ready = True
        return model, tokenizer

    def train(self):
        if (not self.is_data_ready) or (not self.is_model_ready):
            raise RuntimeError('Data and Model must be ready before training!')

        train_loader = DataLoader(
            dataset=self.train_data,
            batch_size=self.mini_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.optimizer_config.lr,
                                      weight_decay=self.optimizer_config.weight_decay,
                                      betas=self.optimizer_config.betas,
                                      eps=self.optimizer_config.eps)
        self.model.train()
        epoch_steps = int(floor(len(train_loader) / self.grad_accumulation_steps))
        total_steps = epoch_steps * self.n_epochs
        eval_steps = int(epoch_steps * 0.1)
        warmup_steps = round(2.0 / (1 - self.optimizer_config.betas[1]))  # https://arxiv.org/pdf/1910.04209

        iteration = 0 # Keeps track of mini-batches
        step = 0 # Keeps track of batches

        lr = get_lr(step=step, min_lr=6e-5, max_lr=6e-4, warmup_iters=warmup_steps, lr_decay_iters=total_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for epoch in range(self.n_epochs):
            print(f'Epoch: {epoch+1} / {self.n_epochs}:')

            # because drop_last = True in dataloader
            optimizer.zero_grad()
            loss_accumulated = 0

            for data_dict in train_loader:
                input_ids = data_dict['input_ids'].to(self.device)
                attention_mask = data_dict['attention_mask'].to(self.device)
                label = data_dict['label'].to(self.device)

                with torch.autocast(device_type=self.device.type):
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = F.cross_entropy(input=logits, target=label, reduction='mean')
                loss = loss / self.grad_accumulation_steps
                loss_accumulated += loss.detach()
                loss.backward()

                if (iteration + 1) % self.grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    loss_accumulated = 0
                    step += 1
                    lr = get_lr(step=step, min_lr=6e-5, max_lr=6e-4, warmup_iters=warmup_steps,
                                lr_decay_iters=total_steps)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                iteration += 1

            print(f'Epoch: {epoch+1} finished')
            train_set_loss = evaluate(model=self.model,
                                      data_loader=DataLoader(dataset=self.train_data,
                                                             batch_size=self.mini_batch_size,
                                                             shuffle=False,
                                                             num_workers=4,
                                                             pin_memory=True,
                                                             drop_last=True))
            valid_set_loss = evaluate(model=self.model,
                                      data_loader=DataLoader(dataset=self.valid_data,
                                                             batch_size=self.mini_batch_size,
                                                             shuffle=False,
                                                             num_workers=4,
                                                             pin_memory=True,
                                                             drop_last=True))
            print(f'\t\tTrain Set Loss : {train_set_loss:.4f}')
            print(f'\t\tValid Set Loss : {valid_set_loss:.4f}')



