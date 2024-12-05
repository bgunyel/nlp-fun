import torch
import torch.nn as nn
from pydantic import BaseModel
from transformers import AutoModel


class ModelConfig(BaseModel):
    backbone: str
    n_backbone_params_to_train: int
    fc_hidden_size: int
    dropout_prob: float


class MachineTranslationModel(nn.Module):
    def __init__(self, config: [ModelConfig, BaseModel], num_classes: int):
        super().__init__()


    def forward(self):
        raise NotImplementedError
