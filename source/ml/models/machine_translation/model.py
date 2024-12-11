import torch
import torch.nn as nn
from pydantic import BaseModel
from transformers import AutoModel


class ModelConfig(BaseModel):
    name: str
    bos_token: str
    eos_token: str


class MachineTranslationModel(nn.Module):
    def __init__(self, vocabulary_size: int, num_classes: int):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.num_classes = num_classes


    def forward(self):
        raise NotImplementedError
