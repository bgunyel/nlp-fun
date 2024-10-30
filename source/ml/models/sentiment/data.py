from __future__ import annotations
from itertools import accumulate

from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from source.ml.datasets import *


class SentimentDataset(Dataset):

    def __init__(self, data_list: list[DatasetBase], data_separations: list[int]):
        self.data_list = data_list
        self.data_separations = data_separations

    def get_info(self) -> list[dict]:
        return [d.get_info() for d in self.data_list]

    @classmethod
    def build_data_splits(cls, tokenizer: [PreTrainedTokenizer, PreTrainedTokenizerFast]):

        base_component = DynaSent.build_data_splits(tokenizer=tokenizer, name='DynaSent', data_round=1)
        bridge_set_proportion = float(len(base_component['bridge'])) / len(base_component['train'])

        data_dicts_list = [
            base_component,
            DynaSent.build_data_splits(tokenizer=tokenizer, name='DynaSent', data_round=2, bridge_set_proportion=bridge_set_proportion),
            SST.build_data_splits(tokenizer=tokenizer, name='SST', bridge_set_proportion=bridge_set_proportion),
        ]
        out_splits = dict()
        for split in ['train', 'bridge', 'validation']:
            data_list = [data_dict[split] for data_dict in data_dicts_list]
            data_separations = [0] + list(accumulate([len(d) for d in data_list]))
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
