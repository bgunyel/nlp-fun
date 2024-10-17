import datetime
import json
import os
import time
from math import log, ceil
from random import randint

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
        self.model_config=ModelConfig(backbone=self.train_config.backbone)
        self.model, self.tokenizer = self.prepare_model()
        print(f'Number of Model Parameters: {self.get_number_of_model_parameters()}')
        self.optimizer = self.configure_optimizers()
        self.train_data, self.valid_data = self.prepare_data()


    def prepare_data(self):
        if not self.is_model_ready:
            raise RuntimeError('Tokenizer must be prepared before training!')

        self.is_data_ready = True
        train_data = SentimentDataset(data_split='train', tokenizer=self.tokenizer)
        valid_data = SentimentDataset(data_split='validation', tokenizer=self.tokenizer)
        return train_data, valid_data

    def prepare_model(self) -> tuple[nn.Module, nn.Module]:
        model = SentimentModel(config=self.model_config, num_classes=self.train_config.n_classes).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.train_config.backbone)
        tokenizer.model_max_length = model.backbone.embeddings.position_embeddings.num_embeddings  # to be on the safe side
        self.is_model_ready = True
        return model, tokenizer

    def train(self):
        if (not self.is_data_ready) or (not self.is_model_ready):
            raise RuntimeError('Data and Model must be ready before training!')

        train_start = datetime.datetime.now().replace(microsecond=0)

        expected_pre_training_loss = -log(1.0 / self.train_config.n_classes)  # -ln(1/n_classes) for cross entropy loss
        initial_train_set_loss, initial_valid_set_loss = self.evaluate()

        print('Pre-Training Stats:')
        print(f'\t\tExpected Loss  : {expected_pre_training_loss:.4f}')
        print(f'\t\tTrain Set Loss : {initial_train_set_loss:.4f}')
        print(f'\t\tValid Set Loss : {initial_valid_set_loss:.4f}')

        train_loader = DataLoader(
            dataset=self.train_data,
            batch_size=self.train_config.mini_batch_size,
            shuffle=True,
            num_workers=settings.NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )

        self.model.train()
        total_steps = int(ceil(self.train_config.n_epochs * len(train_loader) / self.grad_accumulation_steps))
        warmup_steps = round(total_steps * 0.03)
        max_lr = self.optimizer_config.lr
        min_lr = max_lr / 10

        log_train_loss = [-1.0] * self.train_config.n_epochs
        log_valid_loss = [-1.0] * self.train_config.n_epochs
        log_epoch_duration = [-1.0] * self.train_config.n_epochs

        log_batch_loss = [-1.0] * total_steps
        log_grad_norm = [-1.0] * total_steps
        log_lr = [-1.0] * total_steps


        iteration = 0 # Keeps track of mini-batches
        step = 0 # Keeps track of batches

        lr = get_lr(step=step, min_lr=min_lr, max_lr=max_lr, warmup_iters=warmup_steps, lr_decay_iters=total_steps)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        log_lr[step] = lr

        self.optimizer.zero_grad()
        loss_accumulated = 0

        # START TRAINING LOOP
        for epoch in range(self.train_config.n_epochs):
            print(f'Epoch: {epoch+1} / {self.train_config.n_epochs}:')
            epoch_start = time.time()

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
                    norm = torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=1.0, norm_type=2)
                    log_batch_loss[step] = loss_accumulated.item()
                    log_grad_norm[step] = norm.item()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    loss_accumulated = 0

                    # Update for the next step
                    step += 1
                    lr = get_lr(step=step, min_lr=min_lr, max_lr=max_lr, warmup_iters=warmup_steps,
                                lr_decay_iters=total_steps)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    if step < total_steps:
                        log_lr[step] = lr
                    else:
                        print(f'Step: {step}')
                        print(f'Total Steps: {total_steps}')

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

        log_epoch_df = pd.DataFrame(data={
            'train_loss': log_train_loss,
            'valid_loss': log_valid_loss,
        })
        log_step_df = pd.DataFrame(data={
            'batch_loss': log_batch_loss,
            'grad_norm': log_grad_norm,
            'learning_rate': log_lr,
        })

        log_epoch_df.to_parquet(path=os.path.join(settings.OUT_FOLDER, f'train_log__epoch__{train_start}.parquet'))
        log_step_df.to_parquet(path=os.path.join(settings.OUT_FOLDER, f'train_log__step__{train_start}.parquet'))
        self.save_config(config_file_name=os.path.join(settings.OUT_FOLDER, f'config__{train_start}.json'))


    def evaluate(self):
        train_set_loss = evaluate(model=self.model,
                                  data_loader=DataLoader(dataset=self.train_data,
                                                         batch_size=self.train_config.mini_batch_size,
                                                         shuffle=False,
                                                         num_workers=settings.NUM_WORKERS,
                                                         pin_memory=True,
                                                         drop_last=True))
        valid_set_loss = evaluate(model=self.model,
                                  data_loader=DataLoader(dataset=self.valid_data,
                                                         batch_size=self.train_config.mini_batch_size,
                                                         shuffle=False,
                                                         num_workers=settings.NUM_WORKERS,
                                                         pin_memory=True,
                                                         drop_last=True))
        return train_set_loss, valid_set_loss


    def save_config(self, config_file_name: str):

        config_dict = {
            'train_config': self.train_config.model_dump(),
            'optimizer_config': self.optimizer_config.model_dump(),
            'model_config': {
                'fc_hidden_size': self.model_config.fc_hidden_size,
                'dropout_prob': self.model_config.dropout_prob,
            }
        }

        with open(config_file_name, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=4)


    def fit_to_one_batch(self):

        if (not self.is_data_ready) or (not self.is_model_ready):
            raise RuntimeError('Data and Model must be ready before one batch fitting experiment!')

        expected_pre_training_loss = -log(1.0 / self.train_config.n_classes)  # -ln(1/n_classes) for cross entropy loss

        print('--- One Batch Fitting ---')
        print(f'Expected Pre-Training Loss: {expected_pre_training_loss:.4f}')

        train_loader = DataLoader(
            dataset=self.train_data,
            batch_size=self.train_config.mini_batch_size,
            shuffle=True,
            num_workers=settings.NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )

        self.model.train()

        batch_idx = randint(0, len(train_loader) - 1)
        idx = 0

        for data_dict in train_loader:
            if idx == batch_idx:
                print(f'Batch Idx: {idx}')

                for iteration in range(100):
                    input_ids = data_dict['input_ids'].to(self.device)
                    attention_mask = data_dict['attention_mask'].to(self.device)
                    label = data_dict['label'].to(self.device)

                    self.optimizer.zero_grad()

                    with torch.autocast(device_type=self.device.type):
                        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        loss = F.cross_entropy(input=logits, target=label, reduction='mean')

                    loss.backward()
                    self.optimizer.step()
                    print(f'Iteration: {iteration} Loss: {loss.item()}')
            else:
                idx += 1

        print('--- One Batch Fitting Ended ---')
