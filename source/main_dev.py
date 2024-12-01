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
    time_now = datetime.datetime.now().replace(microsecond=0).astimezone(
        tz=datetime.timezone(offset=datetime.timedelta(hours=3), name='UTC+3'))

    print(f'{settings.APPLICATION_NAME} started at {time_now}')
    time1 = time.time()
    main()
    time2 = time.time()
    time_delta = datetime.timedelta(seconds=time2 - time1)

    time_now = datetime.datetime.now().replace(microsecond=0).astimezone(
        tz=datetime.timezone(offset=datetime.timedelta(hours=3), name='UTC+3'))
    print(f'{settings.APPLICATION_NAME} finished at {time_now}')
    print(f'{settings.APPLICATION_NAME} took ' + (str(time_delta)))
