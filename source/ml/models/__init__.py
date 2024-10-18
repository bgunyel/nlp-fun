import importlib
import torch.nn as nn
from pydantic import BaseModel
from source.ml.models.base import TrainerBase, TrainConfig, OptimizerConfig

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

def get_trainer(train_config: TrainConfig, optimizer_config: OptimizerConfig, model_config: BaseModel) -> TrainerBase:
    module = importlib.import_module(name=f'{MODULE_PREFIX}.{train_config.module_name}')
    class_ = getattr(module, 'TheTrainer')
    instance = class_(train_config, optimizer_config, model_config)
    return instance

def get_model_config(module_name: str, params_dict: dict) -> BaseModel:
    module = importlib.import_module(name=f'{MODULE_PREFIX}.{module_name}')
    class_ = getattr(module, 'ModelConfig')
    instance = class_(**params_dict)
    return instance
