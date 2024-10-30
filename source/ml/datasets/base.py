from __future__ import annotations
from abc import abstractmethod
from typing import Optional

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


class DatasetBase(Dataset):
    def __init__(self, name: str, stoi: dict, labels: list, input_ids: list, attention_mask: list):
        super().__init__()
        self.name = name
        self.stoi = stoi
        self.labels = labels
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def get_info(self):
        return {'name': self.name, 'length': len(self) }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
        attention_mask = torch.tensor(self.attention_mask[idx], dtype=torch.long)
        label = self.stoi[self.labels[idx]]
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': label}

    @classmethod
    def build_data_splits(cls, tokenizer: [PreTrainedTokenizer, PreTrainedTokenizerFast],
                          name: str, data_round: Optional[int] = None,
                          bridge_set_proportion: Optional[float] = None) -> dict[str, DatasetBase]:

        valid_input_ids, valid_attention_mask, valid_labels, stoi = cls.prepare_encodings(data_round=data_round,
                                                                                    data_split='validation',
                                                                                    tokenizer=tokenizer)
        input_ids, attention_mask, labels, _ = cls.prepare_encodings(data_round=data_round,
                                                                  data_split='train',
                                                                  tokenizer=tokenizer)

        if bridge_set_proportion is None:
            bridge_set_proportion = float(len(valid_labels)) / len(labels)

        (
            train_input_ids,
            bridge_input_ids,
            train_attention_mask,
            bridge_attention_mask,
            train_labels,
            bridge_labels,
        ) = train_test_split(input_ids, attention_mask, labels,
                             test_size=bridge_set_proportion, random_state=1881, shuffle=True, stratify=labels)

        if data_round is not None:
            name = name + f'-Round-{data_round}'

        return {
            'train': cls(name=name,
                         stoi=stoi,
                         labels=train_labels,
                         input_ids=train_input_ids,
                         attention_mask=train_attention_mask),
            'bridge': cls(name=name,
                          stoi=stoi,
                          labels=bridge_labels,
                          input_ids=bridge_input_ids,
                          attention_mask=bridge_attention_mask),
            'validation': cls(name=name,
                              stoi=stoi,
                              labels=valid_labels,
                              input_ids=valid_input_ids,
                              attention_mask=valid_attention_mask)
        }


    @classmethod
    @abstractmethod
    def prepare_encodings(cls, tokenizer: [PreTrainedTokenizer, PreTrainedTokenizerFast],
                          data_split: str, data_round: int = None) -> tuple[list, list, list, dict]:
        pass



