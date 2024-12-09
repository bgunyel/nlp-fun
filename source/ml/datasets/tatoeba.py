from __future__ import annotations
import os

import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


class Tatoeba(Dataset):
    def __init__(self, source_sentences: list[str], target_sentences: list[str],
                 source_language: str, target_language: str,
                 source_tokenizer: [PreTrainedTokenizer, PreTrainedTokenizerFast],
                 target_tokenizer: [PreTrainedTokenizer, PreTrainedTokenizerFast]) -> None:
        super().__init__()
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.source_language = source_language
        self.target_language = target_language
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer

    def get_info(self):
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.source_sentences)

    def __getitem__(self, idx):

        source_sentence = self.source_sentences[idx]
        target_sentence = self.target_sentences[idx]

        source_encoding = self.source_tokenizer.encode_plus(
            text=source_sentence,
            max_length=self.source_tokenizer.model_max_length,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        target_encoding = self.target_tokenizer.encode_plus(
            text=target_sentence,
            max_length=self.target_tokenizer.model_max_length,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )

        out = {
            'source': {
                'language': self.source_language,
                'text': source_sentence,
                'input_ids': torch.tensor(source_encoding['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(source_encoding['attention_mask'], dtype=torch.long)
            },
            'target': {
                'language': self.target_language,
                'text': target_sentence,
                'input_ids': torch.tensor(target_encoding['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(target_encoding['attention_mask'], dtype=torch.long)
            }
        }

        return out


    @classmethod
    def build_dataset(cls, dataset_folder: str, dataset_split: str, source_language: str, target_language: str,
                      source_tokenizer: [PreTrainedTokenizer, PreTrainedTokenizerFast],
                      target_tokenizer: [PreTrainedTokenizer, PreTrainedTokenizerFast]) -> Tatoeba:

        if dataset_split in ['valid', 'validation']:
            dataset_split = 'dev'

        language_code = f'{source_language}-{target_language}'

        with open(os.path.join(dataset_folder, language_code, f'{dataset_split}.src')) as f:
            source_sentences = f.read().splitlines()
        with open(os.path.join(dataset_folder, language_code, f'{dataset_split}.trg')) as f:
            target_sentences = f.read().splitlines()

        if len(source_sentences) != len(target_sentences):
            message = (f'Number of source ({len(source_sentences)}) & '
                       f'target ({len(target_sentences)}) sentences do NOT match')
            raise RuntimeError(message)

        return cls(source_sentences=source_sentences,
                   target_sentences=target_sentences,
                   source_language=source_language,
                   target_language=target_language,
                   source_tokenizer=source_tokenizer,
                   target_tokenizer=target_tokenizer)
