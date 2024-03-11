import json, os, shutil
from argparse import ArgumentParser
from datetime import datetime

import torch


def get_args():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    current_time = datetime.now()
    current_time = '{:%Y_%m_%d_%H:%M:%S}.{:04.0f}'.format(current_time, 
                                                          current_time.microsecond / 10000.0)
    dir_results = os.path.join("results", current_time)

    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dataset", type=str, default="soccernet_dataset")
    parser.add_argument("--lr", str=float, default=0.1, help="initial lr")
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dir_resume_ckpt", type=str, default=None)
    parser.add_argument("--job_description", type=str, default=None)
    parser.add_argument("--save_ckpt_every_n_epochs", type=int, default=10)
    parser.add_argument("--save_last_ckpt", action="store_true")
    parser.add_argument("--validation_every_n_epochs", type=int, default=10)


    args = parser.parse_args()
    dir_results = f"{dir_results}_B{args.batch_size}_E{args.max_epochs}"
    args.dir_results = dir_results
    args.output_file = os.path.join(dir_results, "output.log")
    args.dir_tb = os.path.join(dir_results, "tb")
    args.dir_ckpt = os.path.join(dir_results, "ckpt")
    args.device = device
    args.current_time = current_time

    # make directories
    os.makedirs("results", exist_ok=True)
    os.mkdir(args.dir_results)
    os.mkdir(args.dir_tb)
    os.mkdir(args.dir_ckpt)
    os.mkdir(os.path.join(dir_results, "data"))

    # save the model architecture
    shutil.copy("data/soccernet_dataset.py", os.path.join(dir_results, "data"))
    shutil.copytree("model/", os.path.join(dir_results, "model"))
    shutil.copytree("train/", os.path.join(dir_results, "train"))
    shutil.copytree("utils/", os.path.join(dir_results, "utils"))

    # dump configuration file
    with open(os.path.join(args.dir_results, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    return args

