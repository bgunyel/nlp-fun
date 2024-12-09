import os
import time
import datetime
import json
from math import log, ceil
from random import randint

import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from source.config import settings
from source.ml.models.base import TrainerBase, TrainConfig, OptimizerConfig
from source.ml.datasets.tatoeba import Tatoeba
from source.ml.utils import evaluate, get_lr
from .model import MachineTranslationModel, ModelConfig


class TheTrainer(TrainerBase):
    def __init__(self, train_config: TrainConfig, optimizer_config: OptimizerConfig, model_config: ModelConfig):
        super().__init__(train_config=train_config, optimizer_config=optimizer_config, model_config=model_config)
        self.name = 'Machine Translation'
        self.tokenizer_names = {
            'epo': 'google-t5/t5-base',
            'eng': 'google-t5/t5-base',
            'tur': 'boun-tabi-LMG/TURNA'
        }
        self.source_tokenizer, self.target_tokenizer, self.forward_model, self.backward_model = self.prepare_model()
        self.forward_optimizer = self.configure_optimizer(model=self.forward_model)
        self.backward_optimizer = self.configure_optimizer(model=self.backward_model)
        self.train_data, self.valid_data = self.prepare_data()
        self.print_info()


    def print_info(self):
        print(f'Trainer Name: {self.name}')
        print(f'Train Set: {self.train_data.get_info()}')
        print(f'Validation Set: {self.valid_data.get_info()}')
        print(f'Model Config: {self.model_config}')
        print(f'Number of Model Parameters: {self.get_number_of_model_parameters():,}')
        print(f'Train Config: {self.train_config}')
        print(f'Optimizer Config: {self.optimizer_config}')
        print(f'Number of Workers: {settings.NUM_WORKERS}')


    def prepare_data(self):
        if not self.is_model_ready:
            raise RuntimeError('Tokenizer must be prepared before training!')

        self.is_data_ready = True

        train_data = Tatoeba.build_dataset(dataset_folder=os.path.join(settings.DATA_FOLDER, 'tatoeba'),
                                           source_language=self.train_config.source_language,
                                           target_language=self.train_config.target_language,
                                           dataset_split='train',
                                           source_tokenizer=self.source_tokenizer,
                                           target_tokenizer=self.target_tokenizer,
                                           bos_token=self.model_config.bos_token,
                                           eos_token=self.model_config.eos_token)

        valid_data = Tatoeba.build_dataset(dataset_folder=os.path.join(settings.DATA_FOLDER, 'tatoeba'),
                                           source_language=self.train_config.source_language,
                                           target_language=self.train_config.target_language,
                                           dataset_split='valid',
                                           source_tokenizer=self.source_tokenizer,
                                           target_tokenizer=self.target_tokenizer,
                                           bos_token=self.model_config.bos_token,
                                           eos_token=self.model_config.eos_token)

        return train_data, valid_data


    def prepare_model(self) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:

        source_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_names[self.train_config.source_language])
        target_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_names[self.train_config.target_language])
        source_tokenizer.model_max_length = 1024  # to be on the safe side
        target_tokenizer.model_max_length = 1024  # to be on the safe side
        source_tokenizer.add_tokens(new_tokens=[self.model_config.bos_token, self.model_config.eos_token])
        target_tokenizer.add_tokens(new_tokens=[self.model_config.bos_token, self.model_config.eos_token])

        forward_model = MachineTranslationModel(vocabulary_size=source_tokenizer.vocab_size, num_classes=target_tokenizer.vocab_size)
        backward_model = MachineTranslationModel(vocabulary_size=target_tokenizer.vocab_size, num_classes=source_tokenizer.vocab_size)
        self.is_model_ready = True

        return source_tokenizer, target_tokenizer, forward_model, backward_model

    def adjust_lr(self, iteration: int, min_lr: float, max_lr: float, warmup_iterations: int, n_iterations: int):
        lr = get_lr(step=iteration, min_lr=min_lr, max_lr=max_lr, warmup_iters=warmup_iterations,
                    lr_decay_iters=n_iterations)
        for param_group in self.forward_optimizer.param_groups:
            param_group['lr'] = lr

        for param_group in self.backward_optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):
        if (not self.is_data_ready) or (not self.is_model_ready):
            raise RuntimeError('Data and Model must be ready before training!')

        train_start = datetime.datetime.now().replace(microsecond=0).astimezone(
            tz=datetime.timezone(offset=datetime.timedelta(hours=3), name='UTC+3'))

        train_loader = DataLoader(
            dataset=self.train_data,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            num_workers=settings.NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )

        self.model.train()
        n_iterations = int(ceil(self.train_config.n_epochs * len(train_loader)))
        warmup_iterations = round(n_iterations * 0.03)
        max_lr = self.optimizer_config.lr
        min_lr = max_lr / 10

        iteration = 0

        # START TRAINING LOOP
        for epoch in range(self.train_config.n_epochs):
            print(f'Epoch: {epoch + 1} / {self.train_config.n_epochs}:')
            epoch_start = time.time()

            for data_dict in train_loader:
                source_ids = data_dict['source']['input_ids'].to(self.device)
                target_ids = data_dict['target']['input_ids'].to(self.device)
                source_attention_count = data_dict['source']['attention_count'].to(self.device)
                target_attention_count = data_dict['target']['attention_count'].to(self.device)

                self.adjust_lr(iteration=iteration, min_lr=min_lr, max_lr=max_lr,
                               warmup_iterations=warmup_iterations, n_iterations=n_iterations)

                self.forward_optimizer.zero_grad()
                self.backward_optimizer.zero_grad()


                iteration += 1

            # End of epoch
            epoch_finish = time.time()
            epoch_duration = epoch_finish - epoch_start


        dummy = -43


    def evaluate(self):
        raise NotImplementedError


    def save_config(self, config_file_name: str):

        config_dict = {
            'train_config': self.train_config.model_dump(),
            'optimizer_config': self.optimizer_config.model_dump(),
            'model_config': self.model_config.model_dump()
        }

        with open(config_file_name, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=4)


    def fit_to_one_batch(self):
        raise NotImplementedError
