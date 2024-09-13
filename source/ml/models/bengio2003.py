from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken
from tqdm import tqdm

from source.config import settings


@dataclass
class ModelConfig:
    block_size: int  # length of input token sequence
    vocabulary_size: int
    embeddings_dim: int  # length of embedding vectors
    n_hidden_units: int
    n_output_units: int


@dataclass
class TrainConfig:
    n_epochs: int
    batch_size: int


class TokenDataset(Dataset):
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


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.block_size = config.block_size
        self.vocabulary_size = config.vocabulary_size
        self.embeddings_dim = config.embeddings_dim

        self.embedding = nn.Embedding(num_embeddings=config.vocabulary_size, embedding_dim=config.embeddings_dim)

        self.hidden = nn.Sequential(
            nn.Linear(in_features=config.embeddings_dim * config.block_size, out_features=config.n_hidden_units),
            nn.Tanh()
        )
        self.out_layer = nn.Linear(in_features=config.n_hidden_units, out_features=config.n_output_units)


    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x_emb = self.embedding(idx)
        x = self.hidden(x_emb.view(-1, self.block_size * self.embeddings_dim))
        logits = self.out_layer(x)
        return logits


def train(device: torch.device):
    with open(settings.BROWN_FILE, 'r') as f:
        text = f.read()

    words = text.split()

    n_train_words = 800000
    n_validation_words = 200000

    tokenizer = tiktoken.get_encoding('cl100k_base')

    train_words = words[:n_train_words]
    valid_words = words[n_train_words + 1: n_train_words + n_validation_words]
    test_words = words[n_train_words + n_validation_words + 1:]

    BLOCK_SIZE = 8
    BLANK_TOKEN_ID = tokenizer.n_vocab
    VOCAB_SIZE = BLANK_TOKEN_ID + 1

    train_data = TokenDataset(words=train_words, block_size=BLOCK_SIZE, tokenizer=tokenizer)
    valid_data = TokenDataset(words=valid_words, block_size=BLOCK_SIZE, tokenizer=tokenizer)

    model_config = ModelConfig(
        block_size=BLOCK_SIZE,
        vocabulary_size=VOCAB_SIZE,
        embeddings_dim=64,
        n_hidden_units=128,
        n_output_units=VOCAB_SIZE
    )
    train_config = TrainConfig(n_epochs=5, batch_size=32)

    train_loader = DataLoader(train_data, batch_size=train_config.batch_size, shuffle=True, num_workers=4)

    model = MLP(config=model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=5e-4,
                                  weight_decay=1e-2,
                                  betas=(0.9, 0.99),
                                  eps=1e-8)

    for epoch in range(train_config.n_epochs):

        tqdm_train_loader = tqdm(train_loader, unit="batch", desc=f'Epoch: {epoch}')

        for iteration, (x, y) in enumerate(tqdm_train_loader):

            x = x.to(device)
            y = y.to(device)

            """
            x_list = []
            y_list = []

            for _ in range(BLOCK_SIZE):
                x_list.append(x)
                y_list.append(y)
                y = x[:, -1]
                x = torch.roll(x, 1, 1)
                x[:, 0] = BLANK_TOKEN_ID  # special <BLANK> token

            x_grand = torch.concat(tensors=x_list, dim=0).to(device)
            y_grand = torch.concat(tensors=y_list, dim=0).to(device)
            """

            logits = model(x)
            loss = F.cross_entropy(input=logits, target=y)

            model.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if iteration % 1000 == 0:
                print(f"Iteration {iteration} | loss {loss.item():.4f} ")


            dummy = -32

    dummy = -32
