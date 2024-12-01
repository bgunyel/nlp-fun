import os
import importlib
import time

from source.config import settings
from source.ml.models.base import TrainerBase, TrainConfig, OptimizerConfig
from source.ml.datasets import Tatoeba
from .model import ModelConfig


class TheTrainer(TrainerBase):
    def __init__(self, train_config: TrainConfig, optimizer_config: OptimizerConfig, model_config: ModelConfig):
        super().__init__(train_config=train_config, optimizer_config=optimizer_config, model_config=model_config)
        self.name = 'Tokenizer'
        self.model = self.prepare_model()
        self.train_data = self.prepare_data()

    def prepare_data(self):
        dataset = Tatoeba.build_dataset(dataset_folder=os.path.join(settings.DATA_FOLDER, 'tatoeba'),
                                        dataset_split='test',
                                        source_language=self.train_config.source_language,
                                        target_language=self.train_config.target_language)
        out = {
            self.train_config.source_language: ' '.join(dataset.get_source_sentences()),
            self.train_config.target_language: ' '.join(dataset.get_target_sentences())
        }
        return out

    def prepare_model(self):
        module = importlib.import_module(name='minbpe')
        class_ = getattr(module, self.model_config.name)

        out = dict()
        if (self.train_config.source_language is None) and (self.train_config.target_language is None):
            out['default'] = class_()
        else:
            if self.train_config.source_language is not None:
                out[self.train_config.source_language] = class_()
            if self.train_config.target_language is not None:
                out[self.train_config.target_language] = class_()

        return out

    def train(self):

        if self.train_config.vocabulary_size % 1000 == 0:
            vocab_size_identifier = f'{round(self.train_config.vocabulary_size / 1000)}k'
        else:
            vocab_size_identifier = f'{round(self.train_config.vocabulary_size / 1000, 2)}k'

        for key, tokenizer in self.model.items():
            print(f'Tokenizer: {key}')
            time1 = time.time()
            tokenizer.train(text=self.train_data[key], vocab_size=self.train_config.vocabulary_size)
            time2 = time.time()
            print(f'Tokenizer: {key} took {time2 - time1} seconds')
            save_name = f'{self.train_config.dataset_name}-{key}_{vocab_size_identifier}'
            tokenizer.save(os.path.join(settings.OUT_FOLDER, save_name))


    def evaluate(self):
        raise NotImplementedError

    def fit_to_one_batch(self):
        raise NotImplementedError