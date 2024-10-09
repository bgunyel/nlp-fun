from itertools import accumulate

from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from source.ml.datasets import *


class SentimentDataset(Dataset):

    def __init__(self, data_split: str, tokenizer: [PreTrainedTokenizer, PreTrainedTokenizerFast]):
        self.data = []
        self.data.append(DynaSent(data_round=1, data_split=data_split, tokenizer=tokenizer))
        self.data.append(DynaSent(data_round=2, data_split=data_split, tokenizer=tokenizer))
        
        self.data_separations = [0] + list(accumulate([d.__len__() for d in self.data]))

    def __len__(self) -> int:
        return self.data_separations[-1]

    def __getitem__(self, idx):
        out = None
        for i in range(1, len(self.data_separations)):
            if idx < self.data_separations[i]:
                out = self.data[i-1].__getitem__(idx - self.data_separations[i-1])
                break
        return out
