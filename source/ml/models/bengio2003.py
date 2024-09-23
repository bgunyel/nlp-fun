from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken
from tqdm import tqdm

from source.config import settings
from source.ml.utils import evaluate


@dataclass
class ModelConfig:
    block_size: int = 8  # length of input token sequence
    embeddings_dim: int = 64  # length of embedding vectors
    n_hidden_units: int = 128

@dataclass
class OptimizerConfig:
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8

@dataclass
class TrainConfig:
    n_epochs: int = 5
    batch_size: int = 32


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
    def __init__(self, config: ModelConfig, vocabulary_size: int, num_classes: int):
        super().__init__()

        self.block_size = config.block_size
        self.embeddings_dim = config.embeddings_dim
        self.vocabulary_size = vocabulary_size

        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=config.embeddings_dim)

        self.hidden = nn.Sequential(
            nn.Linear(in_features=config.embeddings_dim * config.block_size, out_features=config.n_hidden_units),
            nn.Tanh()
        )
        self.out_layer = nn.Linear(in_features=config.n_hidden_units, out_features=num_classes)


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

    train_config = TrainConfig()
    model_config = ModelConfig()
    optimizer_config = OptimizerConfig()

    # BLANK_TOKEN_ID = tokenizer.n_vocab
    # VOCAB_SIZE = BLANK_TOKEN_ID + 1
    VOCAB_SIZE = tokenizer.n_vocab

    train_data = TokenDataset(words=train_words, block_size=model_config.block_size, tokenizer=tokenizer)
    valid_data = TokenDataset(words=valid_words, block_size=model_config.block_size, tokenizer=tokenizer)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    model = MLP(config=model_config, vocabulary_size=VOCAB_SIZE, num_classes=VOCAB_SIZE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=optimizer_config.lr,
                                  weight_decay=optimizer_config.weight_decay,
                                  betas=optimizer_config.betas,
                                  eps=optimizer_config.eps)
                                  
    model.train()
    epoch_iters = len(train_loader)
    eval_iters = int(epoch_iters * 0.1)

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

            optimizer.zero_grad()

            with torch.autocast(device_type=device.type):
                logits = model(x)
                loss = F.cross_entropy(input=logits, target=y)

            loss.backward()
            optimizer.step()

            if (iteration % eval_iters == 0) or (iteration == epoch_iters - 1):

                train_set_loss = evaluate(
                    model=model,
                    data_loader=DataLoader(dataset=train_data,
                                           batch_size=train_config.batch_size,
                                           shuffle=False,
                                           num_workers=4,
                                           pin_memory=True,
                                           drop_last=False)
                )

                valid_set_loss = evaluate(
                    model=model,
                    data_loader=DataLoader(dataset=valid_data,
                                           batch_size=train_config.batch_size,
                                           shuffle=False,
                                           num_workers=4,
                                           pin_memory=True,
                                           drop_last=False)
                )

                print(
                    f"\tEpoch {epoch}\t Iteration {iteration}\t "
                    f"Batch Loss {loss.item():.4f} | "
                    f"Train Set Loss {train_set_loss:.4f} | "
                    f"Valid Set Loss {valid_set_loss:.4f} "
                )


            dummy = -32

    dummy = -32
