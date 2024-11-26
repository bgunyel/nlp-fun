import torch
import torch.nn as nn
from pydantic import BaseModel
from transformers import AutoModel


class ModelConfig(BaseModel):
    backbone: str
    n_backbone_params_to_train: int
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

        n_total_backbone_params = len(list(self.backbone.named_parameters()))
        n_backbone_params_to_train = min(n_total_backbone_params, config.n_backbone_params_to_train)

        # freeze backbone model parameters
        for idx, (name, param) in enumerate(self.backbone.named_parameters()):
            if idx < n_total_backbone_params - n_backbone_params_to_train:
                param.requires_grad = False
            else:
                param.requires_grad = True

        dummy = -32


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        backbone_out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        x = backbone_out.last_hidden_state[:, 0, :]
        # x = torch.mean(input=backbone_out.last_hidden_state, dim=1, keepdim=False)

        out = self.dropout(x)
        out = self.dense(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.out_fc(out)
        return out  # Logits
