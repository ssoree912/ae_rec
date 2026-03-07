import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from rectified.datasets import RealSRPairDataset
from rectified.rectifier_unet import RectifierUNet


def parse_args():
    p = argparse.ArgumentParser(description="Train real-only rectifier R(SR(x)) -> x")
    p.add_argument("--real_root", type=str, required=True, help="Path to train/0_real")
    p.add_argument("--sr_root", type=str, required=True, help="Path to SR cache for train/0_real")
    p.add_argument("--save_path", type=str, required=True, help="Checkpoint output path")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--base_channels", type=int, default=32)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(args):
    ds = RealSRPairDataset(
        real_root=args.real_root,
        sr_root=args.sr_root,
        image_size=args.image_size,
    )
    n_val = max(1, int(round(len(ds) * args.val_ratio)))
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )
    return train_loader, val_loader


def evaluate(model, loader, loss_fn, device):
    model.eval()
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for x, x_sr, _ in loader:
            x = x.to(device, non_blocking=True)
            x_sr = x_sr.to(device, non_blocking=True)
            x_hat = model(x_sr)
            loss = loss_fn(x_hat, x)
            b = x.shape[0]
            loss_sum += float(loss.item()) * b
            count += b
    return loss_sum / max(1, count)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    train_loader, val_loader = build_loaders(args)

    model = RectifierUNet(in_channels=3, base_channels=args.base_channels).to(device)
    loss_fn = nn.L1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    print(f"[Rectifier] train_batches={len(train_loader)} val_batches={len(val_loader)} device={device}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        running = 0.0
        seen = 0
        for x, x_sr, _ in pbar:
            x = x.to(device, non_blocking=True)
            x_sr = x_sr.to(device, non_blocking=True)

            x_hat = model(x_sr)
            loss = loss_fn(x_hat, x)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            b = x.shape[0]
            running += float(loss.item()) * b
            seen += b
            pbar.set_postfix({"train_l1": f"{running / max(1, seen):.5f}"})

        val_l1 = evaluate(model, val_loader, loss_fn, device)
        train_l1 = running / max(1, seen)
        print(f"[Rectifier] epoch={epoch} train_l1={train_l1:.6f} val_l1={val_l1:.6f}")

        ckpt = {
            "model": model.state_dict(),
            "model_cfg": {"in_channels": 3, "base_channels": args.base_channels},
            "epoch": epoch,
            "train_l1": train_l1,
            "val_l1": val_l1,
            "args": vars(args),
        }
        torch.save(ckpt, args.save_path.replace(".pth", "_last.pth"))
        if val_l1 < best_val:
            best_val = val_l1
            torch.save(ckpt, args.save_path)
            print(f"[Rectifier] saved best -> {args.save_path}")


if __name__ == "__main__":
    main()

