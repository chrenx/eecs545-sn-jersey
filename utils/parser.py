import json, os, shutil
from argparse import ArgumentParser
from datetime import datetime

import torch


def get_args():
    device = "cpu"
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        device = "cuda"
    current_time = datetime.now()
    current_time = '{:%Y_%m_%d_%H:%M:%S}.{:04.0f}'.format(current_time, 
                                                          current_time.microsecond / 10000.0)
    dir_results = os.path.join("results", current_time)

    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset", type=str, default="soccernet_dataset")
    parser.add_argument("--max_epochs", type=int, default=300)

    args = parser.parse_args()
    args.dir_tb = os.path.join(dir_results, "tb")
    args.dir_ckpt = os.path.join(dir_results, "ckpt")
    dir_results = f"{args.dir_results}_B{args.batch_size}_E{args.max_epochs}"

    # make directories
    os.makedirs("results", exist_ok=True)
    os.mkdir(args.dir_results)
    os.mkdir(args.dir_tb)
    os.mkdir(args.dir_ckpt)

    # save the model architecture
    shutil.copytree("data/soccernet_dataset.py", os.path.join(dir_results, "data"))
    shutil.copytree("model/", os.path.join(dir_results, "model"))
    shutil.copytree("train/", os.path.join(dir_results, "train"))
    shutil.copytree("utils/", os.path.join(dir_results, "utils"))

    # dump configuration file
    with open(os.path.join(args.dir_results, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

