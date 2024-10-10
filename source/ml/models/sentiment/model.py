from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModel


@dataclass
class ModelConfig:
    backbone: str
    fc_hidden_size: int = 0  # TODO: Currently not used
    dropout_prob: float = 0 # TODO: Currently not used


class SentimentModel(nn.Module):
    def __init__(self, config: ModelConfig, num_classes: int):
        super().__init__()
        self.name = 'Sentiment Analysis'
        self.backbone = AutoModel.from_pretrained(config.backbone)
        self.backbone_hidden_size = self.backbone.config.hidden_size
        # self.fc_hidden_size = config.fc_hidden_size
        # self.dropout_prob = config.dropout_prob

        # self.dense = nn.Linear(self.backbone_hidden_size, self.fc_hidden_size)
        # self.activation = nn.ReLU()
        # self.dropout = nn.Dropout(self.dropout_prob)
        # self.out_fc = nn.Linear(self.fc_hidden_size, num_classes)

        self.out_fc = nn.Linear(self.backbone_hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        backbone_out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        x = backbone_out.last_hidden_state[:, 0, :]

        # out = self.dropout(x)
        # out = self.dense(out)
        # out = self.activation(out)
        # out = self.dropout(out)
        out = self.out_fc(x)
        return out  # Logits
