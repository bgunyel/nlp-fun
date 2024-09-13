import os
import time
import datetime

import torch

from config import settings
import source.ml.models.bengio2003 as bengio2003


def main():
    print(torch.__version__)
    if torch.cuda.is_available():
        print(f'CUDA Current Device: {torch.cuda.current_device()}')
    else:
        raise RuntimeError('No GPU found!')

    print(f'Data Folder: {settings.DATA_FOLDER}')
    print(f'Out Folder: {settings.OUT_FOLDER}')

    device = torch.device('cuda')

    bengio2003.train(device=device)







    dummy = -32


if __name__ == '__main__':
    print(f'{settings.APPLICATION_NAME} started at {datetime.datetime.now()}')
    time1 = time.time()
    main()
    time2 = time.time()
    print(f'{settings.APPLICATION_NAME} finished at {datetime.datetime.now()}')
    print(f'{settings.APPLICATION_NAME} took {(time2 - time1):.2f} seconds')
