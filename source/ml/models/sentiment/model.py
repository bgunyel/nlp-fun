import torch
import torch.nn as nn
from pydantic import BaseModel
from transformers import AutoModel


class ModelConfig(BaseModel):
    backbone: str
    n_backbone_layers_to_train: int
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


        n_total_backbone_layers = len(self.backbone.encoder.layer)
        """
        n_layers_to_train = min(max(0, config.n_backbone_layers_to_train), n_total_backbone_layers)
        layer_ids = []
        if n_layers_to_train > 0:
            layer_ids = list(range(n_total_backbone_layers - n_layers_to_train, n_total_backbone_layers))
        """
        layer_ids = [n_total_backbone_layers - 1]
        # sub_strings = [f'encoder.layer.{i}.intermediate' for i in layer_ids] + ['pooler'] + [f'encoder.layer.{i}.output' for i in layer_ids]
        sub_strings = ['pooler'] + [f'encoder.layer.{i}.output' for i in layer_ids]

        # freeze base model parameters
        for name, param in self.backbone.named_parameters():
            if any(x in name for x in sub_strings):
                param.requires_grad = True
            else:
                param.requires_grad = False

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
