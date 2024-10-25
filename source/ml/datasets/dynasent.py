from __future__ import annotations

from datasets import load_dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from .base import DatasetBase


class DynaSent(DatasetBase):
    def __init__(self, name: str, stoi: dict, labels: list, input_ids: list, attention_mask: list):
        super().__init__(name=name, stoi=stoi, labels=labels, input_ids=input_ids, attention_mask=attention_mask)

    @classmethod
    def prepare_encodings(cls, tokenizer: [PreTrainedTokenizer, PreTrainedTokenizerFast],
                          data_split: str, data_round: int = None) -> tuple[list, list, list, dict]:

        if data_round is None:
            raise ValueError('data_round must be provided (currently None')

        name = f'dynabench.dynasent.r{data_round}.all'
        data_dict = load_dataset(path='dynabench/dynasent', name=name, trust_remote_code=True)
        data = data_dict[data_split]
        encodings = tokenizer.batch_encode_plus(
            data['sentence'],
            max_length=tokenizer.model_max_length,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        stoi = {'negative': 0, 'neutral': 1, 'positive': 2}
        return encodings['input_ids'], encodings['attention_mask'], data['gold_label'], stoi



