import torch

from source.ml.models import get_trainer
from source.ml.models.base import TrainConfig


def train_fine_tune(device: torch.device):

    train_config = TrainConfig(
        model_name='sentiment',
        backbone='prajjwal1/bert-mini',
        max_length=512,
        dataset_names=['SetFit/sst5', 'dynabench/dynasent'],
        n_epochs=1,
        batch_size=32
    )

    trainer = get_trainer(train_config=train_config)

    dummy = -32





