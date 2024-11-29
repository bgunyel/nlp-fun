import os
import tomlkit

from source.config import settings
from source.ml.models import get_trainer, get_model_config
from source.ml.models.base import TrainConfig, OptimizerConfig


def train_fine_tune():

    module_name = 'sentiment'
    config_file_path = os.path.join(settings.INPUT_FOLDER, 'config.toml')

    with open(config_file_path, 'rb') as f:
        config_data = tomlkit.load(f)
        config_data[module_name]['train_config']['module_name'] = module_name

    train_config = TrainConfig(**config_data[module_name]['train_config'])
    optimizer_config = OptimizerConfig(**config_data[module_name]['optimizer_config'])
    model_config = get_model_config(module_name=module_name, params_dict=config_data[module_name]['model_config'])

    if (train_config.batch_size is not None) and (train_config.mini_batch_size is not None):
        if train_config.batch_size % train_config.mini_batch_size != 0:
            raise ValueError('mini_batch_size must be divisible by batch_size')

    trainer = get_trainer(train_config=train_config, optimizer_config=optimizer_config, model_config=model_config)
    # trainer.fit_to_one_batch()
    trainer.train()
