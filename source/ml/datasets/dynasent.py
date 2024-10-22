from __future__ import annotations

from sklearn.model_selection import train_test_split
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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
        attention_mask = torch.tensor(self.attention_mask[idx], dtype=torch.long)
        label = self.stoi[self.labels[idx]]
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': label}

    @classmethod
    def build_data_splits(cls, data_round: int, tokenizer: [PreTrainedTokenizer, PreTrainedTokenizerFast]) -> dict[str, DynaSent]:

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

        return {
            'train': cls(name=f'Round-{data_round}',
                         labels=train_labels,
                         input_ids=train_input_ids,
                         attention_mask=train_attention_mask),
            'bridge': cls(name=f'Round-{data_round}',
                          labels=bridge_labels,
                          input_ids=bridge_input_ids,
                          attention_mask=bridge_attention_mask),
            'validation': cls(name=f'Round-{data_round}',
                              labels=valid_labels,
                              input_ids=valid_input_ids,
                              attention_mask=valid_attention_mask)
        }


    @classmethod
    def prepare_encodings(cls, data_round: int, data_split: str,
                          tokenizer: [PreTrainedTokenizer, PreTrainedTokenizerFast]) -> tuple[list, list, list]:
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
        return encodings['input_ids'], encodings['attention_mask'], data['gold_label']



