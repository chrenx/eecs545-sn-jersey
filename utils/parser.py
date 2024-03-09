import os
from argparse import ArgumentParser
from datetime import datetime

import torch


def get_args():
    os.makedirs("results", exist_ok=True)

    device = "cpu"
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        device = "cuda"
    current_time = datetime.now()
    current_time = '{:%Y_%m_%d_%H:%M:%S}.{:04.0f}'.format(current_time, 
                                                          current_time.microsecond / 10000.0)
    result_dir = os.path.join("results", current_time)
    parser = ArgumentParser()

    args = parser.parse_args()

    os.mkdir(args.result_dir)

