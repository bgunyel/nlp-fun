import os
from torch.utils.data import Dataset


class Tatoeba(Dataset):
    def __init__(self, dataset_folder: str, dataset_split: str):
        super().__init__()

        with open(os.path.join(dataset_folder, f'{dataset_split}.src')) as f:
            src_lines = f.read().splitlines()

        with open(os.path.join(dataset_folder, f'{dataset_split}.trg')) as f:
            trg_lines = f.read().splitlines()

        dummy = -32

    def get_info(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
