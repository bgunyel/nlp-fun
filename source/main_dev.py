import datetime
import os
import time

import torch

from source.config import settings
from source.ml.train_fine_tune import train_fine_tune


def main():
    print(torch.__version__)
    if torch.cuda.is_available():
        print(f'CUDA Current Device: {torch.cuda.current_device()}')
    else:
        raise RuntimeError('No GPU found!')

    print(f'Data Folder: {settings.DATA_FOLDER}')
    print(f'Out Folder: {settings.OUT_FOLDER}')

    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    torch.manual_seed(seed=1881)

    # train_from_scratch(device=torch.device(model_settings.DEVICE))
    train_fine_tune()


if __name__ == '__main__':
    print(f'{settings.APPLICATION_NAME} started at {datetime.datetime.now().replace(microsecond=0)}')
    time1 = time.time()
    main()
    time2 = time.time()
    print(f'{settings.APPLICATION_NAME} finished at {datetime.datetime.now().replace(microsecond=0)}')
    print(f'{settings.APPLICATION_NAME} took {(time2 - time1):.2f} seconds')
