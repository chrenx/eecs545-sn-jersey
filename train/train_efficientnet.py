import importlib, logging, os, sys

import numpy as np
import torch
from timeit import default_timer as timer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from data.custom_dataset import CustomDataset
from model.efficientnet import EfficientNetV2
from utils.parser import get_args

logging.basicConfig(stream=sys.stderr, level=logging.INFO)


def train_efficientnet(args):
    model = EfficientNetV2()
    print(model)
    logging.info("********************** job start time: " + args.current_time)
    logging.info(f"Batch Size: {args.batch_size}, Epoch: {args.max_epochs}")
    
    if args.job_description is not None:
        print("#################################################################")
        print("Job DESCRIPTION:")
        print(args.job_description)
        print("#################################################################\n")
    
    #! Process Previous Training ===============================================
    args.tb_writer = SummaryWriter(args.tb_dir)

    hag3d_module = importlib.import_module("model.hag3d_" + args.hag3d_version)
    HAG3D = getattr(hag3d_module, 'HAG3D')

    hag3d_model = HAG3D(args).to(args.device)

    optimizer = hag3d_model.get_optimizer()
    epoch_start = 0

    if args.resume_ckpt_dir is not None:
        print("Resume training...", args.resume_ckpt_dir)
        ckpt = torch.load(args.resume_ckpt_dir)
        hag3d_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        epoch_start = ckpt['epoch']
        args.current_epoch = epoch_start

        # save previous model
        if args.save_ckpt:
            save_dir=os.path.join(args.ckpt_dir, "pretrained_model.pth")
            torch.save(
                {
                    'epoch': args.current_epoch,
                    'model_state_dict': hag3d_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, 
                save_dir
            )
    else:
        print("Training from begining")

    #! LOAD data ===============================================================
    train_dataset = CustomDataset(args, mode="train")
    test_dataset = CustomDataset(args, mode="test")

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers
    )
    logging.info("train_dataloader done...\n")

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers
    )
    logging.info("test_dataloader done...\n")

    logging.info(f"Length of train dataloader: {len(train_dataloader)} batches of {args.batch_size}")
    logging.info(f"Length of test dataloader: {len(test_dataloader)} batches of {args.batch_size}\n")

    start_train_time = timer()
    args.offset_time = 0

    # print(hag3d_model)

    #! START Training ==========================================================
    #! START Training ==========================================================
    print("Model is on device:", next(hag3d_model.parameters()).device)
    print("\nTraining starts ==========================================\n")

    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    args.early_stopper = EarlyStopper(args.no_early_stop)

    with tqdm(range(epoch_start, args.max_epochs)) as max_epochs:

        args.printout = max_epochs

        # for epoch in range(epoch_start, max_epochs):

        # for epoch in range(epoch_start, max_epochs):
        for epoch in max_epochs:
            # args.printout.set_description(f"Epoch {epoch}")
            # args.printout.set_description(f"Epoch {epoch}")
            args.current_epoch = epoch

            #! TRAIN ===========================================================
            train_step(model=hag3d_model,
                       data_loader=train_dataloader,
                       optimizer=optimizer,
                       args=args)

            #! TEST ============================================================
            if (epoch + 1) % args.validation_every_n_epochs == 0:
                test_step(model=hag3d_model,
                        data_loader=test_dataloader,
                        args=args)
 
            #! SAVE CKPT =======================================================
            if args.early_stopper.stop_needed:
                save_ckpt(hag3d_model, args, optimizer, must_save=True)
                break
            elif args.save_ckpt:
                save_ckpt(hag3d_model, args, optimizer)

    args.tb_writer.close()
    end_train_time = timer()
    print_train_time(start_train_time, end_train_time, 
                     args.device, args.offset_time)

    print("\nTraining ends ============================================\n")


if __name__ == "__main__":
    args = get_args()
    train_efficientnet(args)