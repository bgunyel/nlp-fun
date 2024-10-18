from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModel
from pydantic import BaseModel


class ModelConfig(BaseModel):
    backbone: str
    fc_hidden_size: int
    dropout_prob: float


class SentimentModel(nn.Module):
    def __init__(self, config: [ModelConfig, BaseModel], num_classes: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(config.backbone)
        self.backbone_hidden_size = self.backbone.config.hidden_size
        self.fc_hidden_size = config.fc_hidden_size
        self.dropout_prob = config.dropout_prob

        self.dense = nn.Linear(self.backbone_hidden_size, self.fc_hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_prob)
        self.out_fc = nn.Linear(self.fc_hidden_size, num_classes)

        # self.out_fc = nn.Linear(self.backbone_hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        backbone_out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        x = backbone_out.last_hidden_state[:, 0, :]
        # x = torch.mean(input=backbone_out.last_hidden_state, dim=1, keepdim=False)

        # out = self.dropout(x)
        out = self.dense(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.out_fc(out)
        return out  # Logits
