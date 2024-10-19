from __future__ import annotations
from itertools import accumulate

from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from source.ml.datasets import *


class SentimentDataset(Dataset):

    def __init__(self, data_list: list[Dataset], data_separations: list[int]):
        self.data_list = data_list
        self.data_separations = data_separations

    @classmethod
    def build_dataset(cls, data_split: str, tokenizer: [PreTrainedTokenizer, PreTrainedTokenizerFast]) -> SentimentDataset:
        data_list = [
            DynaSent.build_dataset(data_round=1, data_split=data_split, tokenizer=tokenizer),
            DynaSent.build_dataset(data_round=2, data_split=data_split, tokenizer=tokenizer)
        ]
        data_separations = [0] + list(accumulate([d.__len__() for d in data_list]))
        return cls(data_list=data_list, data_separations=data_separations)

    @classmethod
    def build_data_splits(cls, tokenizer: [PreTrainedTokenizer, PreTrainedTokenizerFast]):

        data_dicts_list = [
            DynaSent.build_data_splits(data_round=1, tokenizer=tokenizer),
            DynaSent.build_data_splits(data_round=2, tokenizer=tokenizer),
        ]
        out_splits = dict()
        for split in ['train', 'validation']:
            data_list = [data_dict[split] for data_dict in data_dicts_list]
            data_separations = [0] + list(accumulate([d.__len__() for d in data_list]))
            out_splits[split] = cls(data_list=data_list, data_separations=data_separations)

        return out_splits

    def __len__(self) -> int:
        return self.data_separations[-1]

    def __getitem__(self, idx):
        out = None
        for i in range(1, len(self.data_separations)):
            if idx < self.data_separations[i]:
                out = self.data_list[i-1].__getitem__(idx - self.data_separations[i-1])
                break
        return out
