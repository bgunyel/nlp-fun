from __future__ import annotations

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


class DynaSent(Dataset):
    def __init__(self, name: str, labels: list, input_ids: list, attention_mask: list):
        super().__init__()
        self.name = name
        self.stoi = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.labels = labels
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    @classmethod
    def build_dataset(cls, data_round: int, data_split: str, tokenizer: [PreTrainedTokenizer, PreTrainedTokenizerFast]) -> DynaSent:
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
        return cls(name=name, labels=data['gold_label'], input_ids=encodings['input_ids'], attention_mask=encodings['attention_mask'])

    @classmethod
    def build_data_splits(cls, data_round: int, tokenizer: [PreTrainedTokenizer, PreTrainedTokenizerFast]) -> dict[str, DynaSent]:
        return {
            'train': cls.build_dataset(data_round=data_round, data_split='train', tokenizer=tokenizer),
            'validation': cls.build_dataset(data_round=data_round, data_split='validation', tokenizer=tokenizer),
        }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
        attention_mask = torch.tensor(self.attention_mask[idx], dtype=torch.long)
        label = self.stoi[self.labels[idx]]
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': label}
