from dataclasses import dataclass

import torch
import torch.nn as nn


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


class TheModel(nn.Module):
    def __init__(self, config: ModelConfig, vocabulary_size: int, num_classes: int):
        super().__init__()
        self.name = 'Bengio2003'

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
