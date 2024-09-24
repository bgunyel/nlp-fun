import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from source.config import settings, model_settings
from source.ml.models import bengio2003


@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader):

    device = torch.device(model_settings.DEVICE)
    model.eval()

    losses = torch.zeros(len(data_loader))

    for idx, (x, y) in enumerate(data_loader):
        x = x.to(device)
        y = y.to(device)

        with torch.autocast(device_type=model_settings.DEVICE):
            logits = model(x)
            loss = F.cross_entropy(input=logits, target=y)

        losses[idx] = loss.item()

    model.train()
    return losses.mean()


def get_dataset_splits(dataset_name: str) -> tuple[list, list, list]:

    train_words = None
    valid_words = None
    test_words = None

    match dataset_name:
        case 'brown':
            with open(settings.BROWN_FILE, 'r') as f:
                text = f.read()

            words = text.split()

            n_train_words = 800000
            n_validation_words = 200000

            train_words = words[:n_train_words]
            valid_words = words[n_train_words + 1: n_train_words + n_validation_words]
            test_words = words[n_train_words + n_validation_words + 1:]
        case _:
            pass

    return train_words, valid_words, test_words
