import argparse
import os

import tqdm
import torch
from sklearn.metrics import balanced_accuracy_score

try:
    from tensorboardX import SummaryWriter
except Exception:
    from torch.utils.tensorboard import SummaryWriter

from utils.dataset import add_dataloader_arguments, create_dataloader
from utils.training import EarlyStopping, TrainingModel, add_training_arguments


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DM detector on SR images only (no rectified pipeline)."
    )
    parser.add_argument(
        "--name",
        type=str,
        default="dmidetection_sr",
        help="Experiment name under checkpoints_dir.",
    )
    parser.add_argument(
        "--sr_dataroot",
        type=str,
        required=True,
        help="Path to SR dataset root. Must contain train/ and valid/ splits.",
    )

    parser = add_training_arguments(parser)
    parser = add_dataloader_arguments(parser)
    parser.add_argument(
        "--num_epoches",
        type=int,
        default=1000,
        help="# of epoches at starting learning rate",
    )
    parser.add_argument(
        "--earlystop_epoch",
        type=int,
        default=5,
        help="Number of epochs without loss reduction before lowering the learning rate",
    )

    # Baseline SR experiment defaults.
    parser.set_defaults(use_inversions=True, batched_syncing=True)
    return parser.parse_args()


def main():
    opt = parse_args()
    torch.manual_seed(opt.seed)

    # Force SR-only training source and avoid accidental original/rectified roots.
    opt.dataroot = opt.sr_dataroot

    valid_data_loader = create_dataloader(opt, subdir="valid", is_train=False)
    train_data_loader = create_dataloader(opt, subdir="train", is_train=True)

    print()
    print(f"[SR-Only] dataroot={opt.dataroot}")
    print(f"[SR-Only] use_inversions={opt.use_inversions} batched_syncing={opt.batched_syncing}")
    print("# validation batches = %d" % len(valid_data_loader))
    print("#   training batches = %d" % len(train_data_loader))

    model = TrainingModel(opt, subdir=opt.name)
    writer = SummaryWriter(os.path.join(model.save_dir, "logs"))
    writer_loss_steps = max(1, len(train_data_loader) // 32)
    early_stopping = None
    start_epoch = model.total_steps // max(1, len(train_data_loader))

    for epoch in range(start_epoch, opt.num_epoches + 1):
        if epoch > start_epoch:
            pbar = tqdm.tqdm(train_data_loader)
            for data in pbar:
                loss = model.train_on_batch(data).item()
                pbar.set_description(f"Train loss: {loss:.4f}")
                total_steps = model.total_steps
                if total_steps % writer_loss_steps == 0:
                    writer.add_scalar("train/loss", loss, total_steps)

            model.save_networks(epoch)

        print("Validation ...", flush=True)
        y_true, y_pred, _ = model.predict(valid_data_loader)
        acc = balanced_accuracy_score(y_true, y_pred > 0.0)
        lr = model.get_learning_rate()

        writer.add_scalar("lr", lr, model.total_steps)
        writer.add_scalar("valid/accuracy", acc, model.total_steps)
        print("After {} epoches: val acc = {}".format(epoch, acc), flush=True)

        if early_stopping is None:
            early_stopping = EarlyStopping(
                init_score=acc,
                patience=opt.earlystop_epoch,
                delta=0.001,
                verbose=True,
            )
        else:
            if early_stopping(acc):
                print("Save best model", flush=True)
                model.save_networks("best")
            if early_stopping.early_stop:
                cont_train = model.adjust_learning_rate()
                if cont_train:
                    print("Learning rate dropped by 10, continue training ...", flush=True)
                    early_stopping.reset_counter()
                else:
                    print("Early stopping.", flush=True)
                    break


if __name__ == "__main__":
    main()
