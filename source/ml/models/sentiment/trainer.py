import datetime
import json
import os
import time
from math import floor

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from source.config import settings
from source.ml.models.base import TrainerBase, TrainConfig, OptimizerConfig
from source.ml.utils import evaluate, get_lr

from .model import ModelConfig, SentimentModel
from .data import SentimentDataset


class TheTrainer(TrainerBase):
    def __init__(self, train_config: TrainConfig, optimizer_config: OptimizerConfig):
        super().__init__(train_config=train_config, optimizer_config=optimizer_config)
        self.name = 'Sentiment Analysis'
        self.model_config=ModelConfig(backbone=self.backbone)
        self.model, self.tokenizer = self.prepare_model()
        self.train_data, self.valid_data = self.prepare_data()

    def prepare_data(self):
        if not self.is_model_ready:
            raise RuntimeError('Tokenizer must be prepared before training!')

        self.is_data_ready = True
        train_data = SentimentDataset(data_split='train', tokenizer=self.tokenizer)
        valid_data = SentimentDataset(data_split='validation', tokenizer=self.tokenizer)
        return train_data, valid_data

    def prepare_model(self) -> tuple[nn.Module, nn.Module]:
        model = SentimentModel(config=self.model_config, num_classes=self.n_classes).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.backbone)
        tokenizer.model_max_length = model.backbone.embeddings.position_embeddings.num_embeddings  # to be on the safe side
        self.is_model_ready = True
        return model, tokenizer

    def train(self):
        if (not self.is_data_ready) or (not self.is_model_ready):
            raise RuntimeError('Data and Model must be ready before training!')

        train_start = datetime.datetime.now().replace(microsecond=0)
        log_file_path = os.path.join(settings.OUT_FOLDER, f'train_log__{train_start}.parquet')
        config_file_path = os.path.join(settings.OUT_FOLDER, f'config__{train_start}.json')

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
        # eval_steps = int(epoch_steps * 0.8)
        # n_eval_steps = int(total_steps / eval_steps) + self.n_epochs
        warmup_steps = round(2.0 / (1 - self.optimizer_config.betas[1]))  # https://arxiv.org/pdf/1910.04209

        log_train_loss = [-1.0] * self.n_epochs
        log_valid_loss = [-1.0] * self.n_epochs
        log_epoch_duration = [-1.0] * self.n_epochs


        iteration = 0 # Keeps track of mini-batches
        step = 0 # Keeps track of batches

        lr = get_lr(step=step, min_lr=6e-5, max_lr=6e-4, warmup_iters=warmup_steps, lr_decay_iters=total_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # START TRAINING LOOP
        for epoch in range(self.n_epochs):
            print(f'Epoch: {epoch+1} / {self.n_epochs}:')
            epoch_start = time.time()

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

                    # Update for the next step
                    step += 1
                    lr = get_lr(step=step, min_lr=6e-5, max_lr=6e-4, warmup_iters=warmup_steps,
                                lr_decay_iters=total_steps)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                iteration += 1

            print(f'Epoch: {epoch+1} finished')

            train_set_loss, valid_set_loss = self.evaluate()
            log_train_loss[epoch] = train_set_loss
            log_valid_loss[epoch] = valid_set_loss
            epoch_finish = time.time()
            epoch_duration = epoch_finish - epoch_start
            log_epoch_duration[epoch] = epoch_duration

            print(f'\t\tTrain Set Loss : {train_set_loss:.4f}')
            print(f'\t\tValid Set Loss : {valid_set_loss:.4f}')
            print(f'\t\tEpoch Time     : {epoch_duration:.2f} sec')

        log_df = pd.DataFrame(data={
            'train_loss': log_train_loss,
            'valid_loss': log_valid_loss,
            'epoch_duration': log_epoch_duration,
        })
        log_df.to_parquet(path=log_file_path)
        self.save_config(config_file_name=config_file_path)


    def evaluate(self):
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
        return train_set_loss, valid_set_loss


    def save_config(self, config_file_name: str):

        config_dict = {
            'train_config': {
                'module_name': self.module_name,
                'backbone': self.backbone,
                'n_classes': self.n_classes,
                'n_epochs': self.n_epochs,
                'batch_size': self.mini_batch_size,
                'mini_batch_size': self.mini_batch_size,
            },
            'optimizer_config': {
                'lr': self.optimizer_config.lr,
                'weight_decay': self.optimizer_config.weight_decay,
                'betas': self.optimizer_config.betas,
                'eps': self.optimizer_config.eps,
            },
            'model_config': {
                'fc_hidden_size': self.model_config.fc_hidden_size,
                'dropout_prob': self.model_config.dropout_prob,
            }
        }

        with open(config_file_name, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=4)
