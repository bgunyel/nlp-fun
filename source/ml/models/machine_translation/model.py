import torch
import torch.nn as nn
from pydantic import BaseModel
from transformers import AutoModel


class ModelConfig(BaseModel):
    name: str


class MachineTranslationModel(nn.Module):
    def __init__(self, config: [ModelConfig, BaseModel], num_classes: int):
        super().__init__()


    def forward(self):
        raise NotImplementedError
