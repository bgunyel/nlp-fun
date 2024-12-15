from __future__ import annotations
import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


class Tatoeba(Dataset):
    def __init__(self,
                 source_sentences: list[str],
                 target_sentences: list[str],
                 source_language: str, target_language: str,
                 source_tokenizer: [PreTrainedTokenizer, PreTrainedTokenizerFast],
                 target_tokenizer: [PreTrainedTokenizer, PreTrainedTokenizerFast],
                 bos_token: str,
                 eos_token: str,
                 is_pre_tokenized: bool = False) -> None:
        super().__init__()
        self.is_pre_tokenized = is_pre_tokenized
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.source_language = source_language
        self.target_language = target_language
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer

        (
            self.source_tokenizer_bos_token_id,
            self.source_tokenizer_eos_token_id,
            self.source_tokenizer_pad_token_id
        ) = self.source_tokenizer.convert_tokens_to_ids([bos_token,
                                                         eos_token,
                                                         source_tokenizer.special_tokens_map['pad_token']])

        (
            self.target_tokenizer_bos_token_id,
            self.target_tokenizer_eos_token_id,
            self.target_tokenizer_pad_token_id
        ) = self.target_tokenizer.convert_tokens_to_ids([bos_token,
                                                         eos_token,
                                                         target_tokenizer.special_tokens_map['pad_token']])

        self.source_ids = None
        self.target_ids = None

        if self.is_pre_tokenized:

            source_encodings = self.source_tokenizer.batch_encode_plus(
                source_sentences,
                max_length=self.source_tokenizer.model_max_length,
                add_special_tokens=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='np'
            )

            target_encodings = self.target_tokenizer.batch_encode_plus(
                target_sentences,
                max_length=self.target_tokenizer.model_max_length,
                add_special_tokens=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='np'
            )

            self.source_ids = np.ones_like(source_encodings['input_ids']) * self.source_tokenizer_pad_token_id
            self.source_ids[:, 0] = self.source_tokenizer_bos_token_id
            self.source_ids[:, 1:] = source_encodings['input_ids'][:, :-1]
            source_attention_counts = np.sum(source_encodings['attention_mask'], axis=1) + 1
            self.source_ids[np.arange(len(source_attention_counts)), source_attention_counts] = self.source_tokenizer_eos_token_id

            self.target_ids = np.ones_like(target_encodings['input_ids']) * self.target_tokenizer_pad_token_id
            self.target_ids[:, 0] = self.target_tokenizer_bos_token_id
            self.target_ids[:, 1:] = target_encodings['input_ids'][:, :-1]
            target_attention_counts = np.sum(target_encodings['attention_mask'], axis=1) + 1
            self.target_ids[np.arange(len(target_attention_counts)), target_attention_counts] = self.target_tokenizer_eos_token_id


        dummy = -32



    def get_info(self):
        return {'source': self.source_language, 'target': self.target_language, 'length': len(self) }

    def __len__(self) -> int:
        return len(self.source_sentences)

    def __getitem__(self, idx):

        if self.is_pre_tokenized:
            out = {
                'source': {
                    'input_ids': torch.tensor(data=self.source_ids[idx, :], dtype=torch.long),
                },
                'target': {
                    'input_ids': torch.tensor(data=self.target_ids[idx, :], dtype=torch.long),
                }
            }
        else:

            source_sentence = self.source_sentences[idx]
            target_sentence = self.target_sentences[idx]

            source_encoding = self.source_tokenizer.encode_plus(
                text=source_sentence,
                max_length=self.source_tokenizer.model_max_length,
                add_special_tokens=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True
            )
            target_encoding = self.target_tokenizer.encode_plus(
                text=target_sentence,
                max_length=self.target_tokenizer.model_max_length,
                add_special_tokens=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True
            )

            source_attention_count = sum(source_encoding['attention_mask'])
            target_attention_count = sum(target_encoding['attention_mask'])

            source_ids = [0] * self.source_tokenizer.model_max_length
            source_ids[0] = self.source_tokenizer_bos_token_id
            source_ids[1:source_attention_count+1] = source_encoding['input_ids'][:source_attention_count]
            source_ids[source_attention_count + 1] = self.source_tokenizer_eos_token_id

            target_ids = [0] * self.target_tokenizer.model_max_length
            target_ids[0] = self.target_tokenizer_bos_token_id
            target_ids[1:target_attention_count + 1] = target_encoding['input_ids'][:target_attention_count]
            target_ids[target_attention_count + 1] = self.target_tokenizer_eos_token_id

            out = {
                'source': {
                    'input_ids': torch.tensor(data = source_ids, dtype=torch.long),
                },
                'target': {
                    'input_ids': torch.tensor(data = target_ids, dtype=torch.long ),
                }
            }

        return out


    @classmethod
    def build_dataset(cls,
                      dataset_folder: str,
                      dataset_split: str,
                      source_language: str,
                      target_language: str,
                      source_tokenizer: [PreTrainedTokenizer, PreTrainedTokenizerFast],
                      target_tokenizer: [PreTrainedTokenizer, PreTrainedTokenizerFast],
                      bos_token: str,
                      eos_token: str,
                      is_pre_tokenized: bool = False) -> Tatoeba:

        if dataset_split in ['valid', 'validation']:
            dataset_split = 'dev'

        language_code = f'{source_language}-{target_language}'

        with open(os.path.join(dataset_folder, language_code, f'{dataset_split}.src')) as f:
            source_sentences = f.read().split(sep='\n')[:-1]
        with open(os.path.join(dataset_folder, language_code, f'{dataset_split}.trg')) as f:
            target_sentences = f.read().split(sep='\n')[:-1]

        if len(source_sentences) != len(target_sentences):
            message = (f'Number of source ({len(source_sentences)}) & '
                       f'target ({len(target_sentences)}) sentences do NOT match')
            raise RuntimeError(message)

        return cls(source_sentences=source_sentences,
                   target_sentences=target_sentences,
                   source_language=source_language,
                   target_language=target_language,
                   source_tokenizer=source_tokenizer,
                   target_tokenizer=target_tokenizer,
                   bos_token=bos_token,
                   eos_token=eos_token,
                   is_pre_tokenized=is_pre_tokenized)
