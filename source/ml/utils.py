import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from source.config import model_settings


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