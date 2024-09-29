import torch
from torch.utils.data import Dataset
import tiktoken


class CustomDataset(Dataset):
    def __init__(self, words: list, block_size: int, tokenizer: tiktoken.Encoding):
        self.words = words
        self.block_size = block_size
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.words) - self.block_size

    def __getitem__(self, idx):
        x_words = self.words[idx: idx + self.block_size + 1]
        x_str = ' '.join(x_words)
        x_tokens = self.tokenizer.encode(x_str)

        x = torch.tensor(x_tokens[: self.block_size], dtype=torch.long)  # there may be more tokens than words
        y = torch.tensor(x_tokens[self.block_size], dtype=torch.long)
        return x, y
