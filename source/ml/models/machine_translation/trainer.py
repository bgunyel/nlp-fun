import json
import torch.nn as nn

from source.ml.models.base import TrainerBase, TrainConfig, OptimizerConfig
from .model import MachineTranslationModel, ModelConfig


class TheTrainer(TrainerBase):
    def __init__(self, train_config: TrainConfig, optimizer_config: OptimizerConfig, model_config: ModelConfig):
        super().__init__(train_config=train_config, optimizer_config=optimizer_config, model_config=model_config)
        self.name = 'Machine Translation'


    def print_info(self):
        raise NotImplementedError

    def prepare_data(self):
        raise NotImplementedError

    def prepare_model(self) -> tuple[nn.Module, nn.Module]:
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


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
