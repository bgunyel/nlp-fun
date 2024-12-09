import os
import json
import torch.nn as nn
from transformers import AutoTokenizer

from source.config import settings
from source.ml.models.base import TrainerBase, TrainConfig, OptimizerConfig
from source.ml.datasets.tatoeba import Tatoeba
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
        self.source_tokenizer, self.target_tokenizer = self.prepare_model()
        self.train_data, self.valid_data = self.prepare_data()


    def print_info(self):
        raise NotImplementedError


    def prepare_data(self):
        if not self.is_model_ready:
            raise RuntimeError('Tokenizer must be prepared before training!')

        self.is_data_ready = True

        train_data = Tatoeba.build_dataset(dataset_folder=os.path.join(settings.DATA_FOLDER, 'tatoeba'),
                                           source_language=self.train_config.source_language,
                                           target_language=self.train_config.target_language,
                                           dataset_split='train',
                                           source_tokenizer=self.source_tokenizer,
                                           target_tokenizer=self.target_tokenizer)

        valid_data = Tatoeba.build_dataset(dataset_folder=os.path.join(settings.DATA_FOLDER, 'tatoeba'),
                                           source_language=self.train_config.source_language,
                                           target_language=self.train_config.target_language,
                                           dataset_split='valid',
                                           source_tokenizer=self.source_tokenizer,
                                           target_tokenizer=self.target_tokenizer)

        return train_data, valid_data


    def prepare_model(self) -> tuple[nn.Module, nn.Module]:

        source_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_names[self.train_config.source_language])
        target_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_names[self.train_config.target_language])
        source_tokenizer.model_max_length = 1024  # to be on the safe side
        target_tokenizer.model_max_length = 1024  # to be on the safe side
        self.is_model_ready = True

        return source_tokenizer, target_tokenizer

    def train(self):

        a = self.train_data[0]

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
