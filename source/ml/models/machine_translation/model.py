import torch
import torch.nn as nn
from pydantic import BaseModel


class EncoderConfig(BaseModel):
    embedding_size: int
    hidden_size: int
    num_layers: int
    dropout_prob: float

class ModelConfig(BaseModel):
    name: str
    bos_token: str
    eos_token: str
    encoder: EncoderConfig


class RNNEncoder(nn.Module):
    def __init__(self, config: [EncoderConfig, BaseModel], vocabulary_size: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=config.embedding_size)
        self.rnn = nn.LSTM(input_size=config.embedding_size,
                           hidden_size=config.hidden_size,
                           num_layers=config.num_layers,
                           batch_first=True)
        self.dropout = nn.Dropout(p=config.dropout_prob)

    def forward(self, input_ids: torch.Tensor):
        embedding_output = self.embedding(input_ids)
        x = self.dropout(embedding_output)
        outputs, (hidden, cell) = self.rnn(x)
        return hidden, cell


class RNNEncoderDecoderModel(nn.Module):
    def __init__(self, config: [ModelConfig, BaseModel], vocabulary_size: int, num_classes: int):
        super().__init__()
        self.encoder = RNNEncoder(config=config.encoder, vocabulary_size=vocabulary_size)

    def forward(self, input_ids: torch.Tensor):
        encoder_out = self.encoder(input_ids)
        return encoder_out  # TODO: This will change when Decoder is added


class MachineTranslationModel(nn.Module):
    def __init__(self, config: [ModelConfig, BaseModel], vocabulary_size: int, num_classes: int):
        super().__init__()
        # TODO: Model selection will implemented when multiple models are available
        self.model = RNNEncoderDecoderModel(config=config, vocabulary_size=vocabulary_size, num_classes=num_classes)

        dummy = -32

    def forward(self, input_ids: torch.Tensor):
        out = self.model(input_ids = input_ids)
        return out
