import torch

from source.ml.models import get_trainer
from source.ml.models.base import PreTrainedModelPath, TrainConfig, OptimizerConfig


def train_fine_tune():

    train_config = TrainConfig(
        module_name='sentiment',
        backbone = PreTrainedModelPath.bert_tiny.value,
        dataset_names = ['SetFit/sst5', 'dynabench/dynasent'],
        n_classes = 3,
        n_epochs = 100,
        batch_size = 32,
        mini_batch_size=8,
        device=torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'),
    )

    optimizer_config = OptimizerConfig(
        lr = 3e-4,
        weight_decay = 0.1,
        betas = (0.9, 0.95),
        eps = 1e-8
    )

    if train_config.batch_size % train_config.mini_batch_size != 0:
        raise ValueError('mini_batch_size must be divisible by batch_size')

    trainer = get_trainer(train_config=train_config, optimizer_config=optimizer_config)
    trainer.train()
