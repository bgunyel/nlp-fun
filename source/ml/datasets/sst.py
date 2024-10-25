from __future__ import annotations

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from .base import DatasetBase


class SST(DatasetBase):
    def __init__(self, name: str, labels: list, input_ids: list, attention_mask: list):
        super().__init__(name=name, labels=labels, input_ids=input_ids, attention_mask=attention_mask)

    @classmethod
    def prepare_encodings(cls, tokenizer: [PreTrainedTokenizer, PreTrainedTokenizerFast],
                          data_split: str, data_round: int = None) -> tuple[list, list, list]:
        data_dict = load_dataset(path='SetFit/sst5')
        data = data_dict[data_split]
        encodings = tokenizer.batch_encode_plus(
            data['text'],
            max_length=tokenizer.model_max_length,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        labels = [s.split(" ")[-1] for s in data['label_text']] # convert from 5 labels to 3 labels
        return encodings['input_ids'], encodings['attention_mask'], labels



