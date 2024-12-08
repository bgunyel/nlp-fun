import importlib
import os

from tokenizers import Tokenizer
from tokenizers.decoders import BPEDecoder
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from source.config import settings
from source.ml.models.base import TrainerBase, TrainConfig, OptimizerConfig
from .model import ModelConfig, TokenizationModel


class TheTrainer(TrainerBase):
    def __init__(self, train_config: TrainConfig, optimizer_config: OptimizerConfig, model_config: ModelConfig):
        super().__init__(train_config=train_config, optimizer_config=optimizer_config, model_config=model_config)
        self.name = 'Tokenizer'
        self.tokenizer, self.trainer = self.prepare_model()
        self.model = TokenizationModel()
        self.train_data = self.prepare_data()

    def prepare_data(self):
        out = self.model.get_train_data(language_code=self.train_config.language)
        return out

    def prepare_model(self):
        tokenizer = Tokenizer(BPE(unk_token="[UNK]", continuing_subword_prefix='##', end_of_word_suffix='##'))

        # Get the pre-tokenizer name from model config and create an instance
        module = importlib.import_module(name='tokenizers.pre_tokenizers')
        class_ = getattr(module, self.model_config.pre_tokenizer)
        tokenizer.pre_tokenizer = class_()

        tokenizer.decoder = BPEDecoder(suffix='##')
        trainer = BpeTrainer(vocab_size=self.train_config.vocabulary_size,
                             special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
                             show_progress=True,
                             continuing_subword_prefix='##',
                             end_of_word_suffix='##')
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