import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from source.config import settings
from source.ml.models.base import TrainerBase, TrainConfig, OptimizerConfig
from .model import ModelConfig


class TheTrainer(TrainerBase):
    def __init__(self, train_config: TrainConfig, optimizer_config: OptimizerConfig, model_config: ModelConfig):
        super().__init__(train_config=train_config, optimizer_config=optimizer_config, model_config=model_config)
        self.name = 'Tokenizer'
        self.tokenizer, self.trainer = self.prepare_model()
        self.train_data = self.prepare_data()

    def prepare_data(self):
        match self.train_config.language:
            case 'tur':
                out = [
                    os.path.join(settings.DATA_FOLDER, 'tatoeba', 'cat-tur', 'train.trg'),
                    os.path.join(settings.DATA_FOLDER, 'tatoeba', 'isl-tur', 'train.trg'),
                    os.path.join(settings.DATA_FOLDER, 'tatoeba', 'deu-tur', 'train.trg'),
                    os.path.join(settings.DATA_FOLDER, 'tatoeba', 'eng-tur', 'train.trg')
                ]
            case _:
                out = []
        return out

    def prepare_model(self):
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=self.train_config.vocabulary_size,
                             special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        return tokenizer, trainer

    def train(self):

        if self.train_config.vocabulary_size % 1000 == 0:
            vocab_size_identifier = f'{round(self.train_config.vocabulary_size / 1000)}k'
        else:
            vocab_size_identifier = f'{round(self.train_config.vocabulary_size / 1000, 2)}k'

        self.tokenizer.train(self.train_data, self.trainer)

        save_name = f'tokenizer_{self.train_config.language}-{vocab_size_identifier}.json'
        self.tokenizer.save(path=os.path.join(settings.OUT_FOLDER, save_name), pretty=True)

    def evaluate(self):
        raise NotImplementedError

    def fit_to_one_batch(self):
        raise NotImplementedError