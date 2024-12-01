from __future__ import annotations
import os
from torch.utils.data import Dataset


class Tatoeba(Dataset):
    def __init__(self, source_sentences: list[str], target_sentences: list[str]):
        super().__init__()
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences

    def get_source_sentences(self) -> list[str]:
        return self.source_sentences

    def get_target_sentences(self) -> list[str]:
        return self.target_sentences

    def get_info(self):
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.source_sentences)

    def __getitem__(self, idx):
        raise NotImplementedError

    @classmethod
    def build_dataset(cls, dataset_folder: str, dataset_split: str, source_language: str, target_language: str) -> Tatoeba:

        language_code = f'{source_language}-{target_language}'

        with open(os.path.join(dataset_folder, language_code, f'{dataset_split}.src')) as f:
            source_sentences = f.read().splitlines()
        with open(os.path.join(dataset_folder, language_code, f'{dataset_split}.trg')) as f:
            target_sentences = f.read().splitlines()

        return cls(source_sentences=source_sentences, target_sentences=target_sentences)

