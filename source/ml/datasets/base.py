from __future__ import annotations
from abc import abstractmethod

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


class DatasetBase(Dataset):
    def __init__(self, name: str, labels: list, input_ids: list, attention_mask: list):
        super().__init__()
        self.name = name
        self.stoi = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.labels = labels
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
        attention_mask = torch.tensor(self.attention_mask[idx], dtype=torch.long)
        label = self.stoi[self.labels[idx]]
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': label}

    @classmethod
    def build_data_splits(cls, tokenizer: [PreTrainedTokenizer, PreTrainedTokenizerFast],
                          name: str, data_round: int = None) -> dict[str, DatasetBase]:

        valid_input_ids, valid_attention_mask, valid_labels = cls.prepare_encodings(data_round=data_round,
                                                                                    data_split='validation',
                                                                                    tokenizer=tokenizer)
        input_ids, attention_mask, labels = cls.prepare_encodings(data_round=data_round,
                                                                  data_split='train',
                                                                  tokenizer=tokenizer)

        (
            train_input_ids,
            bridge_input_ids,
            train_attention_mask,
            bridge_attention_mask,
            train_labels,
            bridge_labels
        ) = train_test_split(input_ids, attention_mask, labels,
                             test_size=len(valid_labels), random_state=1881, shuffle=True, stratify=labels)

        if data_round is not None:
            name = name + f'-Round-{data_round}'

        return {
            'train': cls(name=name,
                         labels=train_labels,
                         input_ids=train_input_ids,
                         attention_mask=train_attention_mask),
            'bridge': cls(name=name,
                          labels=bridge_labels,
                          input_ids=bridge_input_ids,
                          attention_mask=bridge_attention_mask),
            'validation': cls(name=name,
                              labels=valid_labels,
                              input_ids=valid_input_ids,
                              attention_mask=valid_attention_mask)
        }


    @classmethod
    @abstractmethod
    def prepare_encodings(cls, tokenizer: [PreTrainedTokenizer, PreTrainedTokenizerFast],
                          data_split: str, data_round: int = None) -> tuple[list, list, list]:
        pass



