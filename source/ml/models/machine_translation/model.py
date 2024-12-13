import torch
import torch.nn as nn
from pydantic import BaseModel


class EncoderConfig(BaseModel):
    embedding_size: int
    hidden_size: int
    num_layers: int
    dropout_prob: float

class DecoderConfig(BaseModel):
    embedding_size: int
    hidden_size: int
    num_layers: int
    dropout_prob: float

class ModelConfig(BaseModel):
    name: str
    bos_token: str
    eos_token: str
    max_sequence_length: int
    encoder: EncoderConfig
    decoder: DecoderConfig


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
        out, (hidden, cell) = self.rnn(x)
        return hidden, cell


class RNNDecoder(nn.Module):
    def __init__(self, config: [DecoderConfig, BaseModel], vocabulary_size: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=config.embedding_size)
        self.rnn = nn.LSTM(input_size=config.embedding_size,
                           hidden_size=config.hidden_size,
                           num_layers=config.num_layers,
                           batch_first=True)
        self.fc_out = nn.Linear(in_features=config.hidden_size, out_features=vocabulary_size)
        self.dropout = nn.Dropout(p=config.dropout_prob)

    def forward(self, input_ids: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor):
        embedding_output = self.embedding(input_ids)
        x = self.dropout(embedding_output)
        x = torch.unsqueeze(input=x, dim=1)
        out, (hidden, cell) = self.rnn(x, (hidden, cell))
        out = torch.squeeze(input=out)
        logits = self.fc_out(out)
        return logits, hidden, cell


class RNNEncoderDecoderModel(nn.Module):
    def __init__(self,
                 config: [ModelConfig, BaseModel],
                 in_vocabulary_size: int,
                 out_vocabulary_size: int,
                 bos_token_id: int,
                 eos_token_id: int):
        super().__init__()
        self.max_sequence_length = config.max_sequence_length
        self.out_vocabulary_size = out_vocabulary_size
        self.encoder = RNNEncoder(config=config.encoder, vocabulary_size=in_vocabulary_size)
        self.decoder = RNNDecoder(config=config.decoder, vocabulary_size=out_vocabulary_size)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def forward(self, input_ids: torch.Tensor, output_ids: torch.Tensor):
        hidden, cell = self.encoder(input_ids)

        logits = torch.zeros(size=(*output_ids.shape, self.out_vocabulary_size)).to(input_ids.device)
        logits[:, 0, output_ids[:, 0]] = 1 # first token is given (<BOS>)
        decoder_input = output_ids[:, 0]

        for i in range(1, self.max_sequence_length):
            logits[:, i, :], hidden, cell = self.decoder(input_ids=decoder_input, hidden=hidden, cell=cell)
            decoder_input = output_ids[:, i]  # input for the next iteration

        return logits


class MachineTranslationModel(nn.Module):
    def __init__(self,
                 config: [ModelConfig, BaseModel],
                 in_vocabulary_size: int,
                 out_vocabulary_size: int,
                 bos_token_id: int,
                 eos_token_id: int):
        super().__init__()
        # TODO: Model selection will implemented when multiple models are available
        self.model = RNNEncoderDecoderModel(config=config,
                                            in_vocabulary_size=in_vocabulary_size,
                                            out_vocabulary_size=out_vocabulary_size,
                                            bos_token_id=bos_token_id,
                                            eos_token_id=eos_token_id)

        dummy = -32

    def forward(self, input_ids: torch.Tensor, output_ids: torch.Tensor):
        out = self.model(input_ids = input_ids, output_ids = output_ids)
        return out
