import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pandas as pd
import matplotlib.pyplot as plt

from source.config import settings, model_settings


@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader) -> float:

    device = torch.device(model_settings.DEVICE)
    model.eval()

    losses = torch.zeros(len(data_loader))

    for idx, data_dict in enumerate(data_loader):
        input_ids = data_dict['input_ids'].to(device)
        attention_mask = data_dict['attention_mask'].to(device)
        label = data_dict['label'].to(device)

        with torch.autocast(device_type=model_settings.DEVICE):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(input=logits, target=label, reduction='mean')

        losses[idx] = loss.item()

    model.train()
    return losses.mean().item()


# learning rate decay scheduler (cosine with warmup) (from Karpathy nanoGPT)
def get_lr(step: int, min_lr: float, max_lr: float, warmup_iters: int, lr_decay_iters: int) -> float:

    if step < warmup_iters:
        out_lr = min_lr + (max_lr - min_lr) * step / warmup_iters
    elif step > lr_decay_iters:
        out_lr = min_lr
    else:
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (step - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        out_lr = min_lr + coeff * (max_lr - min_lr)

    return out_lr


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


def visualize_logs(file_path: str):

    if '__step__' in file_path:
        x_axis_name = 'steps'
    elif '__epoch__' in file_path:
        x_axis_name = 'epochs'
    else:
        x_axis_name = 'None'

    df = pd.read_parquet(path=file_path)
    n_columns = len(df.columns)
    n_points = len(df)

    if x_axis_name == 'epochs':
        plt.figure(figsize=(18, 8))
        for i, col_name in enumerate(df.columns):
            plt.plot(range(n_points), df[col_name], label=col_name, marker='*')
        plt.grid(visible=True)
        plt.legend()
        plt.xlabel(x_axis_name)
        plt.show()
    else:
        for i, col_name in enumerate(df.columns):
            plt.figure(figsize=(18, 8))
            plt.plot(range(n_points), df[col_name])
            plt.grid(visible=True)
            plt.title(col_name)
            plt.xlabel(x_axis_name)
            plt.show()
