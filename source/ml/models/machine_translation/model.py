import random
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # x = torch.unsqueeze(input=x, dim=1)
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
                 eos_token_id: int,
                 pad_token_id: int):
        super().__init__()
        self.max_sequence_length = config.max_sequence_length
        self.out_vocabulary_size = out_vocabulary_size
        self.encoder = RNNEncoder(config=config.encoder, vocabulary_size=in_vocabulary_size)
        self.decoder = RNNDecoder(config=config.decoder, vocabulary_size=out_vocabulary_size)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def forward(self,
                input_ids: torch.Tensor,
                output_ids: torch.Tensor,
                teacher_forcing_probability: float):

        """
        :param input_ids: size N x L where N is the batch size and L is the maximum sequence length of the tokenizer
        :param output_ids: size N x L where N is the batch size and L is the maximum sequence length of the tokenizer
        :param teacher_forcing_probability: float between 0 and 1
        :return:
        """

        if (teacher_forcing_probability < 0) or (teacher_forcing_probability > 1):
            raise RuntimeError('teacher_forcing_probability must be between 0 and 1')

        hidden, cell = self.encoder(input_ids)

        # input_ids is of size (N, L). If the sequence length is shorter, it is padded.
        logits = torch.zeros(size=(*output_ids.shape, self.out_vocabulary_size)).to(input_ids.device)
        logits[:, :, self.pad_token_id] = 1
        logits[:, 0, output_ids[:, 0]] = 1 # first token is given (<BOS>)
        decoder_input = output_ids[:, 0]

        for i in range(1, self.max_sequence_length):
            decoder_input = decoder_input.view(-1, 1)
            logits[:, i, :], hidden, cell = self.decoder(input_ids=decoder_input, hidden=hidden, cell=cell)

            # input for the next iteration
            if random.random() < teacher_forcing_probability:
                decoder_input = output_ids[:, i]
            else:
                probs = F.softmax(logits[:, i, :], dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                decoder_input = idx_next

        return logits


    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor):
        hidden, cell = self.encoder(input_ids)

        out_ids = torch.zeros_like(input_ids).to(input_ids.device)
        out_ids[:, 0] = self.bos_token_id

        for i in range(1, self.max_sequence_length):
            decoder_input = out_ids[:, i-1]
            logits, hidden, cell = self.decoder(input_ids=decoder_input, hidden=hidden, cell=cell)
            probs = F.softmax(logits[:, i, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            out_ids[:, i] = idx_next.item()

            if idx_next.item() == self.eos_token_id:
                break

        return out_ids




class MachineTranslationModel(nn.Module):
    def __init__(self,
                 config: [ModelConfig, BaseModel],
                 in_vocabulary_size: int,
                 out_vocabulary_size: int,
                 bos_token_id: int,
                 eos_token_id: int,
                 pad_token_id: int):
        super().__init__()
        # TODO: Model selection will implemented when multiple models are available
        self.model = RNNEncoderDecoderModel(config=config,
                                            in_vocabulary_size=in_vocabulary_size,
                                            out_vocabulary_size=out_vocabulary_size,
                                            bos_token_id=bos_token_id,
                                            eos_token_id=eos_token_id,
                                            pad_token_id=pad_token_id)

    def forward(self,
                input_ids: torch.Tensor,
                output_ids: torch.Tensor,
                teacher_forcing_probability: float):
        out = self.model(input_ids = input_ids,
                         output_ids = output_ids,
                         teacher_forcing_probability = teacher_forcing_probability)
        return out
