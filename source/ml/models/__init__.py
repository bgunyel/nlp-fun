import importlib
from dataclasses import dataclass
import torch.nn as nn

from source.ml.models.base import TrainerBase, TrainConfig

# from source.ml.models import bengio2003

MODULE_PREFIX = 'source.ml.models'


def get_model(model_name: str, config, vocabulary_size: int, num_classes: int) -> nn.Module:
    module = importlib.import_module(name=f'{MODULE_PREFIX}.{model_name}')
    class_ = getattr(module, 'TheModel')
    instance = class_(config, vocabulary_size, num_classes)
    return instance

def get_config(model_name: str, config_type: str):
    match config_type:
        case 'model' | 'model_config':
            config_class = 'ModelConfig'
        case 'optimizer' | 'optimizer_config':
            config_class = 'OptimizerConfig'
        case _:
            raise RuntimeError(f'Invalid config_type: {config_type}')
    module = importlib.import_module(name=f'{MODULE_PREFIX}.{model_name}')
    class_ = getattr(module, config_class)
    instance = class_()
    return instance

def get_trainer(train_config: TrainConfig) -> TrainerBase:
    module = importlib.import_module(name=f'{MODULE_PREFIX}.{train_config.model_name}')
    class_ = getattr(module, 'TheTrainer')
    instance = class_(train_config)
    return instance