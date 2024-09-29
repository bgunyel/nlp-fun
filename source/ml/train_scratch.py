import tiktoken
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from source.ml.datasets import CustomDataset
from source.ml.models import get_model, get_config
from source.ml.models.base import TrainConfig
from source.ml.utils import get_dataset_splits, evaluate


def train_from_scratch(device: torch.device):

    train_config = TrainConfig(
        dataset_names = ['brown'],
        model_name = 'bengio2003',
        n_epochs = 1,
        batch_size = 32
    )
    model_config = get_config(model_name=train_config.model_name, config_type='model')
    optimizer_config = get_config(model_name=train_config.model_name, config_type='optimizer')

    tokenizer = tiktoken.get_encoding('cl100k_base')
    # BLANK_TOKEN_ID = tokenizer.n_vocab
    # VOCAB_SIZE = BLANK_TOKEN_ID + 1
    VOCAB_SIZE = tokenizer.n_vocab

    train_words, valid_words, test_words = get_dataset_splits(train_config.dataset_names[0])
    train_data = CustomDataset(words=train_words, block_size=model_config.block_size, tokenizer=tokenizer)
    valid_data = CustomDataset(words=valid_words, block_size=model_config.block_size, tokenizer=tokenizer)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    model = get_model(
        model_name=train_config.model_name,
        config=model_config,
        vocabulary_size=VOCAB_SIZE,
        num_classes=VOCAB_SIZE).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=optimizer_config.lr,
                                  weight_decay=optimizer_config.weight_decay,
                                  betas=optimizer_config.betas,
                                  eps=optimizer_config.eps)

    model.train()
    epoch_iters = len(train_loader)
    eval_iters = int(epoch_iters * 0.1)

    for epoch in range(train_config.n_epochs):
        tqdm_train_loader = tqdm(train_loader, unit="batch", desc=f'Epoch: {epoch}')

        for iteration, (x, y) in enumerate(tqdm_train_loader):

            x = x.to(device)
            y = y.to(device)

            """
            x_list = []
            y_list = []

            for _ in range(BLOCK_SIZE):
                x_list.append(x)
                y_list.append(y)
                y = x[:, -1]
                x = torch.roll(x, 1, 1)
                x[:, 0] = BLANK_TOKEN_ID  # special <BLANK> token

            x_grand = torch.concat(tensors=x_list, dim=0).to(device)
            y_grand = torch.concat(tensors=y_list, dim=0).to(device)
            """

            optimizer.zero_grad()

            with torch.autocast(device_type=device.type):
                logits = model(x)
                loss = F.cross_entropy(input=logits, target=y)

            loss.backward()
            optimizer.step()

            if (iteration % eval_iters == 0) or (iteration == epoch_iters - 1):

                train_set_loss = evaluate(
                    model=model,
                    data_loader=DataLoader(dataset=train_data,
                                           batch_size=train_config.batch_size,
                                           shuffle=False,
                                           num_workers=4,
                                           pin_memory=True,
                                           drop_last=False)
                )

                valid_set_loss = evaluate(
                    model=model,
                    data_loader=DataLoader(dataset=valid_data,
                                           batch_size=train_config.batch_size,
                                           shuffle=False,
                                           num_workers=4,
                                           pin_memory=True,
                                           drop_last=False)
                )

                print(
                    f"\tEpoch {epoch}\t Iteration {iteration}\t "
                    f"Batch Loss {loss.item():.4f} | "
                    f"Train Set Loss {train_set_loss:.4f} | "
                    f"Valid Set Loss {valid_set_loss:.4f} "
                )


            dummy = -32

    dummy = -32
