import importlib, os, sys

import numpy as np
import torch
from timeit import default_timer as timer
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from model.efficientnet import EfficientNetV2
from optimizer.sam import SAM
from utils.parser_util import get_args
from utils.train_util import train_step, test_step, print_train_time, \
                             save_ckpt, get_data_loader


def get_optimizer(args, model):
    """
    SGD or SAM (https://arxiv.org/abs/2010.01412)
    """
    if args.optimizer == 'SGD':
        return torch.optim.SGD(model.parameters(), args.lr,
                               momentum=args.momentum,
                               weight_decay=args.weight_decay)
    elif args.optimizer == "SAM":
        base_optimizer = torch.optim.SGD
        return SAM(model.parameters(), base_optimizer, 
                   lr=args.lr, momentum=args.momentum,
                   weight_decay=args.weight_decay)
    return None


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_efficientnet(args):
    model = EfficientNetV2(args).to(args.device)
    print(model.device)
    print("Job start time: " + args.current_time)
    
    print("Job DESCRIPTION: ------------------------------------------------")
    if args.job_description is not None:
        print(args.job_description)
    print(f"Batch Size: {args.batch_size}, Epoch: {args.max_epochs}")
    
    # #! Process Previous Training ===============================================
    args.tb_writer = SummaryWriter(args.tb_dir)

    optimizer = get_optimizer(args.optimizer)

    if args.resume_ckpt_dir is not None:
        print("Resume training...", args.resume_ckpt_dir)
        ckpt = torch.load(args.resume_ckpt_dir)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        epoch_start = ckpt['epoch']
        args.current_epoch = epoch_start

    else:
        print("Training from begining")

    #! LOAD data ===============================================================
    train_data_loader = get_data_loader(args, mode="train")
    test_data_loader = get_data_loader(args, mode="test")

    start_train_time = timer()
    args.offset_time = 0

    print("Training starts ==========================================")

    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    exit(0)

    # args.early_stopper = EarlyStopper(args.no_early_stop)

    with tqdm(range(epoch_start, args.max_epochs)) as max_epochs:

        args.printout = max_epochs

        for epoch in max_epochs:
            args.printout.set_description(f"Epoch {epoch}")
            args.current_epoch = epoch

            #! TRAIN ===========================================================
            train_step(model=model,
                       data_loader=train_data_loader,
                       optimizer=optimizer,
                       args=args)

            #! TEST ============================================================
            if (epoch + 1) % args.validation_every_n_epochs == 0:
                test_step(model=model,
                        data_loader=test_data_loader,
                        args=args)
 
            #! SAVE CKPT =======================================================
            if args.save_ckpt:
                save_ckpt(model, args, optimizer)

    args.tb_writer.close()
    end_train_time = timer()
    print_train_time(start_train_time, end_train_time, 
                     args.device, args.offset_time)

    print("\nTraining ends ============================================\n")


if __name__ == "__main__":
    args = get_args()

    f = open(args.output_file, 'w')
    sys.stdout = f
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    train_efficientnet(args)
    f.close()

