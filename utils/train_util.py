import os, random
from datetime import timedelta

import torch
from timeit import default_timer as timer
from tqdm.auto import tqdm


def track_loss(results, args, mode):
    step_num = args.current_epoch * args.data_loader_len + args.batch_idx

    args.tb_writer.add_scalar(
        f"{mode} loss", 
        results['loss'],
        step_num
        step_num
    )
    # printout = f" {mode}_loss={results['loss']:.3f}"
    printout = f" {mode}_loss={results['loss']}"
    # printout = f" {mode}_loss={results['loss']:.3f}"
    printout = f" {mode}_loss={results['loss']}"

    iterate_loss = results['weighted_loss_dict']
    if mode == "test":
        iterate_loss = results['loss_dict']

    for key, values in iterate_loss.items():
        content = values.mean().item()
        # if mode == "test":
        #     content = values
        args.tb_writer.add_scalar(
            f"{mode}_{key}", 
            content,
            step_num
            step_num
        )
        # printout += f" | {key}={content:.3f}"
        printout += f" | {key}={content}"
    args.printout.set_postfix_str(printout)

#!==============================================================================
def train_step(model, data_loader, optimizer, args):
    args.printout.set_description(f"Epoch {args.current_epoch}")
    args.printout.set_description(f"Epoch {args.current_epoch}")
    args.data_loader_len = len(data_loader)
    model.train()
    loss_per_batch = {'train_loss':[], 'train_weighted_loss': []}
    for batch_idx, records in enumerate(data_loader):
        if args.train_in_validation and batch_idx == 0:
            args.train_data_records = records
        args.batch_idx = batch_idx
        results = model(records, mode="train")
        optimizer.zero_grad()
        results['loss'].backward()
        optimizer.step()
        # Log Loss =======
        track_loss(results, args, "train")

#!==============================================================================
def test_step(model, data_loader, args):
    args.printout.set_description(f"Validation {args.current_epoch}")
    args.data_loader_len = 8 # len(data_loader)
    model.eval()
    loss_per_batch = {'test_loss':[], 'test_loss_dict': []}
    with torch.no_grad():
        if args.train_in_validation and (args.current_epoch+1) % args.viz_every_n_epochs == 0:
            args.batch_idx = -1
            model(args.train_data_records, mode="test")
        args.epoch_viz_done = False

        for batch_idx in tqdm(range(args.validation_num_batches)):
        # for batch_idx, records in enumerate(tqdm(data_loader)):
            records = next(iter(data_loader))
            args.batch_idx = batch_idx

            results = model(records, mode="test")
            track_loss(results, args, "test")
            if not args.no_early_stop and args.early_stopper.early_stop(results['loss'], "test"):
                print("\nEarly stop at epoch", args.current_epoch, "when testing")
                break
            if args.overfit:
                break
        args.epoch_viz_done = False


def save_ckpt(model, args, optimizer):

    if args.current_epoch == 0 or args.debug:
        return
    do_save = False
    if (args.current_epoch+1) % args.save_ckpt_every_n_epochs == 0:
        do_save = True
    elif args.save_last_ckpt and args.current_epoch == args.max_epochs - 1:
        do_save = True

    if do_save:
        start_time = timer()
        args.printout.set_postfix_str(f"Saving ckpt at epoch {args.current_epoch}")
        save_dir = os.path.join(args.ckpt_dir, f"ckpt_epoch={args.current_epoch}.pth")

        torch.save(
            {
                'epoch': args.current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 
            save_dir
        )
        end_time = timer()
        args.offset_time += (end_time - start_time)


def print_train_time(start_time, end_time, device, offset_time=0):
    """
    offset: e.g. time of inference
    """
    train_time_no_inf = end_time - start_time - offset_time
    hr_min_sec_no_inf = timedelta(seconds=train_time_no_inf)
    train_time_inf = end_time - start_time
    hr_min_sec_inf = timedelta(seconds=train_time_inf)

    print(f"\nVisualization time on (hh:mm:ss) {timedelta(seconds=offset_time)}")
    print(f"\nTrain time on {device} (hh:mm:ss) without inference: {hr_min_sec_no_inf}")
    print(f"\nTrain time on {device} (hh:mm:ss) including inference: {hr_min_sec_inf}")


class EarlyStopper:
    def __init__(self, no_early_stop=False, 
                       train_patience=1500, train_min_delta=0.00, 
                       test_patience=1000, test_min_delta=0.00):
        """
        Check early stop for training mode every batch
        Check early stop for testing mode every epoch
        """
        self.train_patience = train_patience
        self.train_min_delta = train_min_delta
        self.train_counter = 0
        self.min_train_loss = float('inf')

        self.test_patience = test_patience
        self.test_min_delta = test_min_delta
        self.test_counter = 0
        self.min_test_loss = float('inf')
        self.no_early_stop = no_early_stop
        self.stop_needed = False

    def early_stop(self, loss, mode):
        if self.no_early_stop:
            self.stop_needed = False
            return False
        if loss is None:
            self.stop_needed = True
            return True
        if mode == "train":
            if loss < self.min_train_loss:
                self.min_train_loss = loss
                self.train_counter = 0
            elif loss > (self.min_train_loss + self.train_min_delta):
                self.train_counter += 1
                if self.train_counter >= self.train_patience:
                    print("Early stop: exceed training patience")
                    self.stop_needed = True
                    return True
        else: # test
            if loss < self.min_test_loss:
                self.min_test_loss = loss
                self.test_counter = 0
            elif loss > (self.min_test_loss + self.test_min_delta):
                self.test_counter += 1
                if self.test_counter >= self.test_patience:
                    print("Early stop: exceed test patience")
                    self.stop_needed = True
                    return True
        return False



