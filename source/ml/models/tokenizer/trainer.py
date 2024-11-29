import os
import importlib

from source.config import settings
from source.ml.models.base import TrainerBase, TrainConfig, OptimizerConfig
from source.ml.datasets import Tatoeba
from .model import ModelConfig


class TheTrainer(TrainerBase):
    def __init__(self, train_config: TrainConfig, optimizer_config: OptimizerConfig, model_config: ModelConfig):
        super().__init__(train_config=train_config, optimizer_config=optimizer_config, model_config=model_config)
        self.name = 'Tokenizer'
        self.tokenizer = self.prepare_model()
        self.train_data = self.prepare_data()

    def prepare_data(self):
        dataset = Tatoeba(dataset_folder=os.path.join(settings.DATA_FOLDER, 'tatoeba', 'eng-tur'),
                          dataset_split='test')
        dummy = -32


    def prepare_model(self):
        module = importlib.import_module(name='minbpe')
        class_ = getattr(module, self.model_config.name)
        instance = class_()
        return instance

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def fit_to_one_batch(self):
        raise NotImplementedError