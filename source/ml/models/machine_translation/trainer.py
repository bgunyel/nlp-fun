import os
import json
import torch.nn as nn
from tokenizers import Tokenizer

from source.config import settings
from source.ml.models.base import TrainerBase, TrainConfig, OptimizerConfig
from .model import MachineTranslationModel, ModelConfig


class TheTrainer(TrainerBase):
    def __init__(self, train_config: TrainConfig, optimizer_config: OptimizerConfig, model_config: ModelConfig):
        super().__init__(train_config=train_config, optimizer_config=optimizer_config, model_config=model_config)
        self.name = 'Machine Translation'
        self.prepare_model()


    def print_info(self):
        raise NotImplementedError

    def prepare_data(self):
        raise NotImplementedError

    def prepare_model(self) -> tuple[nn.Module, nn.Module]:
        dummy = -32
        target_tokenizer = Tokenizer.from_file(path=os.path.join(settings.INPUT_FOLDER, 'tokenizer_tur-100k.json'))

    def train(self):
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
