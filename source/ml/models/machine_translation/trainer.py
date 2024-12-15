import os
import time
import datetime
import json
from math import log, ceil
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from source.config import settings
from source.ml.models.base import TrainerBase, TrainConfig, OptimizerConfig
from source.ml.datasets.tatoeba import Tatoeba
from source.ml.utils import evaluate, get_lr, cosine_decay
from .model import MachineTranslationModel, ModelConfig


class TheTrainer(TrainerBase):
    def __init__(self, train_config: TrainConfig, optimizer_config: OptimizerConfig, model_config: ModelConfig):
        super().__init__(train_config=train_config, optimizer_config=optimizer_config, model_config=model_config)
        self.name = 'Machine Translation'
        self.tokenizer_names = {
            'epo': 'google-t5/t5-base',
            'eng': 'google-t5/t5-base',
            'ina': 'google-t5/t5-base',
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
        print(f'Number of Forward Model Parameters: {self.get_number_of_model_parameters(model=self.forward_model):,}')
        print(f'Number of Backward Model Parameters: {self.get_number_of_model_parameters(model=self.backward_model):,}')
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
        source_tokenizer.model_max_length = self.model_config.max_sequence_length  # to be on the safe side
        target_tokenizer.model_max_length = self.model_config.max_sequence_length  # to be on the safe side
        source_tokenizer.add_tokens(new_tokens=[self.model_config.bos_token, self.model_config.eos_token])
        target_tokenizer.add_tokens(new_tokens=[self.model_config.bos_token, self.model_config.eos_token])

        (source_tokenizer_bos_token_id,
         source_tokenizer_eos_token_id,
         source_tokenizer_pad_token_id) = source_tokenizer.convert_tokens_to_ids([self.model_config.bos_token,
                                                                                  self.model_config.eos_token,
                                                                                  source_tokenizer.special_tokens_map['pad_token']])
        (target_tokenizer_bos_token_id,
         target_tokenizer_eos_token_id,
         target_tokenizer_pad_token_id) = target_tokenizer.convert_tokens_to_ids([self.model_config.bos_token,
                                                                                  self.model_config.eos_token,
                                                                                  target_tokenizer.special_tokens_map['pad_token']])

        forward_model = MachineTranslationModel(config=self.model_config,
                                                in_vocabulary_size=len(source_tokenizer),
                                                out_vocabulary_size=len(target_tokenizer),
                                                bos_token_id=target_tokenizer_bos_token_id,
                                                eos_token_id=target_tokenizer_eos_token_id,
                                                pad_token_id=target_tokenizer_pad_token_id).to(self.device)
        backward_model = MachineTranslationModel(config=self.model_config,
                                                 in_vocabulary_size=len(target_tokenizer),
                                                 out_vocabulary_size=len(source_tokenizer),
                                                 bos_token_id=source_tokenizer_bos_token_id,
                                                 eos_token_id=source_tokenizer_eos_token_id,
                                                 pad_token_id=source_tokenizer_pad_token_id).to(self.device)
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

        print(f'Training started at {train_start}')

        # -ln(1/n_classes) for cross entropy loss
        expected_pre_training_loss_forward = -log(1.0 / self.target_tokenizer.vocab_size)
        expected_pre_training_loss_backward = -log(1.0 / self.source_tokenizer.vocab_size)

        print(f'Expected pre-training loss (forward model): {expected_pre_training_loss_forward:.4f}')
        print(f'Expected pre-training loss (backward model): {expected_pre_training_loss_backward:.4f}')

        train_loader = DataLoader(
            dataset=self.train_data,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            num_workers=settings.NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )

        self.forward_model.train()
        self.backward_model.train()
        n_iterations = int(ceil(self.train_config.n_epochs * len(train_loader)))
        warmup_iterations = round(n_iterations * 0.03)
        max_lr = self.optimizer_config.lr
        min_lr = max_lr / 10

        print(f'Total number of iterations: {n_iterations:,}')
        print(f'Warmup iterations: {warmup_iterations:,}')


        loop_start_datetime = datetime.datetime.now().replace(microsecond=0).astimezone(
            tz=datetime.timezone(offset=datetime.timedelta(hours=3), name='UTC+3'))
        print(f'Loop started at: {loop_start_datetime}')

        loop_start = time.time()

        # eval_iterations = 2000
        # max_eval_iterations = 250

        buffer_size = self.train_config.log_iterations
        train_loss_buffer_forward = np.ones(buffer_size, dtype=float) * expected_pre_training_loss_forward
        train_loss_buffer_backward = np.ones(buffer_size, dtype=float) * expected_pre_training_loss_backward

        iteration = 0

        # START TRAINING LOOP
        for epoch in range(self.train_config.n_epochs):
            print(f'Epoch: {epoch + 1} / {self.train_config.n_epochs}:')
            epoch_start = time.time()

            for data_dict in train_loader:
                source_ids = data_dict['source']['input_ids'].to(self.device)
                target_ids = data_dict['target']['input_ids'].to(self.device)
                # source_attention_count = data_dict['source']['attention_count'].to(self.device)
                # target_attention_count = data_dict['target']['attention_count'].to(self.device)

                self.adjust_lr(iteration=iteration, min_lr=min_lr, max_lr=max_lr,
                               warmup_iterations=warmup_iterations, n_iterations=n_iterations)

                self.forward_optimizer.zero_grad()
                self.backward_optimizer.zero_grad()

                teacher_probability = cosine_decay(iteration=iteration, min_value=0, max_value=1, decay_iters=n_iterations)

                with torch.autocast(device_type=self.device.type):
                    logits_forward = self.forward_model(input_ids=source_ids,
                                                        output_ids=target_ids,
                                                        teacher_forcing_probability=teacher_probability)
                    logits_backward = self.backward_model(input_ids=target_ids,
                                                          output_ids=source_ids,
                                                          teacher_forcing_probability=teacher_probability)
                    loss_forward = F.cross_entropy(input=logits_forward.view(-1, logits_forward.size(-1)),
                                                   target=target_ids.view(-1),
                                                   reduction='mean')
                    loss_backward = F.cross_entropy(input=logits_backward.view(-1, logits_backward.size(-1)),
                                                    target=source_ids.view(-1),
                                                    reduction='mean')

                train_loss_buffer_forward[iteration % buffer_size] = loss_forward.item()
                train_loss_buffer_backward[iteration % buffer_size] = loss_backward.item()

                loss_forward.backward()
                loss_backward.backward()

                norm_forward = torch.nn.utils.clip_grad_norm_(parameters=self.forward_model.parameters(),
                                                              max_norm=1.0, norm_type=2)
                norm_backward = torch.nn.utils.clip_grad_norm_(parameters=self.forward_model.parameters(),
                                                               max_norm=1.0, norm_type=2)

                self.forward_optimizer.step()
                self.backward_optimizer.step()

                if iteration % self.train_config.log_iterations == 0:

                    eval_time_start = time.time()
                    valid_loss_forward, valid_loss_backward = self.evaluate(
                        max_eval_iterations = self.train_config.max_eval_iterations
                    )
                    eval_time_finish = time.time()

                    print(f'iteration: {iteration+1} / {n_iterations:,}: ')
                    print(f'\t eval time: {eval_time_finish-eval_time_start:.2f} seconds')

                    # Logging


                    print(f'\t train loss forward: {train_loss_buffer_forward.mean():.4f} -- '
                          f'valid loss forward: {valid_loss_forward:.4f} -- '
                          f'train loss backward: {train_loss_buffer_backward.mean():.4f} -- '
                          f'valid loss backward: {valid_loss_backward:.4f}')

                if iteration % random.choice([2, 3, 5, 7, 11, 13, 17, 19]) == 0:
                    iteration_finish = time.time()
                    average_iteration_time = (iteration_finish - loop_start) / (iteration + 1)
                    expected_loop_time = average_iteration_time * n_iterations
                    expected_train_end = loop_start_datetime + datetime.timedelta(seconds=expected_loop_time)

                    print(f'iteration: {iteration+1} / {n_iterations:,} ({((iteration+1) / n_iterations * 100):.2f} %) finished '
                          f'@ {average_iteration_time:.2f} seconds/iteration -- '
                          f'Expected Train End: {expected_train_end}')

                # Update iteration index
                iteration += 1

            # End of epoch
            valid_loss_forward, valid_loss_backward = self.evaluate()
            epoch_finish = time.time()
            epoch_duration = epoch_finish - epoch_start


        dummy = -43


    @torch.no_grad()
    def evaluate(self, max_eval_iterations: int = None):

        self.forward_model.eval()
        self.backward_model.eval()

        valid_loader = DataLoader(
            dataset=self.valid_data,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            num_workers=settings.NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )

        if max_eval_iterations is not None:
            max_iterations = int(min(max_eval_iterations, len(valid_loader)))
        else:
            max_iterations = len(valid_loader)

        losses_forward = torch.zeros(max_iterations)
        losses_backward = torch.zeros(max_iterations)

        idx = 0
        for data_dict in valid_loader:
            source_ids = data_dict['source']['input_ids'].to(self.device)
            target_ids = data_dict['target']['input_ids'].to(self.device)

            with torch.autocast(device_type=self.device.type):
                logits_forward = self.forward_model(input_ids=source_ids,
                                                    output_ids=target_ids,
                                                    teacher_forcing_probability=0)
                logits_backward = self.backward_model(input_ids=target_ids,
                                                      output_ids=source_ids,
                                                      teacher_forcing_probability=0)
                loss_forward = F.cross_entropy(input=logits_forward.view(-1, logits_forward.size(-1)),
                                               target=target_ids.view(-1),
                                               reduction='mean')
                loss_backward = F.cross_entropy(input=logits_backward.view(-1, logits_backward.size(-1)),
                                                target=source_ids.view(-1),
                                                reduction='mean')
            losses_forward[idx] = loss_forward.item()
            losses_backward[idx] = loss_backward.item()

            idx += 1
            if idx >= max_iterations:
                break

        self.forward_model.train()
        self.backward_model.train()
        return losses_forward[:idx].mean().item(), losses_backward[:idx].mean().item()


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
