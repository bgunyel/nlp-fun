import torch

from source.ml.models import get_trainer
from source.ml.models.base import TrainConfig, OptimizerConfig


def train_fine_tune():

    train_config = TrainConfig(
        module_name='sentiment',
        backbone = 'prajjwal1/bert-mini',
        max_length = 512,
        dataset_names = ['SetFit/sst5', 'dynabench/dynasent'],
        n_epochs = 1,
        batch_size = 32,
        device=torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'),
    )

    optimizer_config = OptimizerConfig(
        lr = 3e-4,
        weight_decay = 0.1,
        betas = (0.9, 0.95),
        eps = 1e-8
    )

    trainer = get_trainer(train_config=train_config, optimizer_config=optimizer_config)

    dummy = -32





