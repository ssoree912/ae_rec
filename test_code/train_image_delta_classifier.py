import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
TRAINING_CODE_DIR = (THIS_DIR.parent / "training_code").resolve()
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(TRAINING_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_CODE_DIR))

from rectified.sr_binary_pair_dataset import SRBinaryPairDataset  # noqa: E402
from models.delta_classifier import SmallCNN  # noqa: E402
from rectified.rectifier_unet import load_rectifier  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser("Train classifier on image-space delta")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--sr_cache_root", type=str, default=None, help="Optional. Omit for SR-free mode.")
    p.add_argument("--rect_ckpt", type=str, required=True)
    p.add_argument("--save_path", type=str, required=True)

    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true")

    p.add_argument("--rect_input", type=str, default="sr", choices=["sr", "orig"])
    p.add_argument(
        "--delta_mode",
        type=str,
        default="sr_minus_rectified",
        choices=["sr_minus_rectified", "orig_minus_rectified"],
    )

    p.add_argument("--use_abs", action="store_true")
    p.add_argument("--delta_norm", type=str, default="none", choices=["none", "tanh", "clamp"])
    p.add_argument("--delta_scale", type=float, default=3.0)
    p.add_argument("--delta_clip", type=float, default=1.0)

    p.add_argument("--concat_input", action="store_true", help="Use concat(x,delta) as 6-channel classifier input.")
    p.add_argument("--clf_width", type=int, default=64)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def build_delta(
    x,
    x_sr,
    rectifier,
    rect_input="sr",
    delta_mode="sr_minus_rectified",
    use_abs=False,
    delta_norm="none",
    delta_scale=3.0,
    delta_clip=1.0,
):
    rect_in = x_sr if rect_input == "sr" else x
    x_hat = rectifier(rect_in)

    if delta_mode == "sr_minus_rectified":
        delta = x_sr - x_hat
    elif delta_mode == "orig_minus_rectified":
        delta = x - x_hat
    else:
        raise ValueError(delta_mode)

    if use_abs:
        delta = delta.abs()

    if delta_norm == "tanh":
        delta = torch.tanh(delta_scale * delta)
    elif delta_norm == "clamp":
        delta = torch.clamp(delta, -delta_clip, delta_clip) / max(delta_clip, 1e-8)

    return delta


def make_clf_input(x, delta, concat_input=False):
    if concat_input:
        return torch.cat([x, delta], dim=1)
    return delta


@torch.no_grad()
def evaluate_epoch(clf, rectifier, loader, device, args):
    clf.eval()
    rectifier.eval()
    ys, ps = [], []
    loss_sum, count = 0.0, 0

    for x, x_sr, y, _ in loader:
        x = x.to(device, non_blocking=True)
        x_sr = x_sr.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).unsqueeze(1)

        delta = build_delta(
            x,
            x_sr,
            rectifier,
            rect_input=args.rect_input,
            delta_mode=args.delta_mode,
            use_abs=args.use_abs,
            delta_norm=args.delta_norm,
            delta_scale=args.delta_scale,
            delta_clip=args.delta_clip,
        )
        clf_in = make_clf_input(x, delta, args.concat_input)
        logits = clf(clf_in)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        prob = torch.sigmoid(logits).squeeze(1)
        ys.extend(y.squeeze(1).cpu().numpy().tolist())
        ps.extend(prob.cpu().numpy().tolist())

        b = x.size(0)
        loss_sum += float(loss.item()) * b
        count += b

    ys = np.array(ys)
    ps = np.array(ps)
    ap = average_precision_score(ys, ps) if len(np.unique(ys)) > 1 else float("nan")
    auroc = roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else float("nan")
    acc = ((ps > 0.5) == ys).mean()
    return loss_sum / max(1, count), ap, auroc, acc


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    torch.backends.cudnn.benchmark = True

    ds = SRBinaryPairDataset(
        data_root=args.data_root,
        sr_cache_root=args.sr_cache_root,
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
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(max(1, args.num_workers // 2) > 0),
    )

    rectifier = load_rectifier(args.rect_ckpt, device)
    for p in rectifier.parameters():
        p.requires_grad_(False)
    rectifier.eval()

    c_in = 6 if args.concat_input else 3
    clf = SmallCNN(c_in=c_in, width=args.clf_width).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_ap = -1.0
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    print(
        f"[Classifier] rect_input={args.rect_input} delta_mode={args.delta_mode} "
        f"concat_input={args.concat_input} sr_cache_root={args.sr_cache_root}"
    )
    print(f"[Classifier] train_batches={len(train_loader)} val_batches={len(val_loader)} device={device}")

    for epoch in range(1, args.epochs + 1):
        clf.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        run_loss, seen = 0.0, 0

        for x, x_sr, y, _ in pbar:
            x = x.to(device, non_blocking=True)
            x_sr = x_sr.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).unsqueeze(1)

            with torch.no_grad():
                delta = build_delta(
                    x,
                    x_sr,
                    rectifier,
                    rect_input=args.rect_input,
                    delta_mode=args.delta_mode,
                    use_abs=args.use_abs,
                    delta_norm=args.delta_norm,
                    delta_scale=args.delta_scale,
                    delta_clip=args.delta_clip,
                )
                clf_in = make_clf_input(x, delta, args.concat_input)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                logits = clf(clf_in)
                loss = F.binary_cross_entropy_with_logits(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            b = x.size(0)
            run_loss += float(loss.item()) * b
            seen += b
            pbar.set_postfix(train_bce=f"{run_loss / max(1, seen):.5f}")

        val_loss, val_ap, val_auroc, val_acc = evaluate_epoch(clf, rectifier, val_loader, device, args)
        print(f"[Val] epoch={epoch} bce={val_loss:.6f} ap={val_ap:.6f} auroc={val_auroc:.6f} acc@0.5={val_acc:.6f}")

        ckpt = {
            "clf": clf.state_dict(),
            "clf_cfg": {"c_in": c_in, "width": args.clf_width},
            "args": vars(args),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_ap": val_ap,
            "val_auroc": val_auroc,
            "val_acc": val_acc,
        }
        torch.save(ckpt, args.save_path.replace(".pth", "_last.pth"))
        if val_ap > best_ap:
            best_ap = val_ap
            torch.save(ckpt, args.save_path)
            print(f"[Classifier] saved best -> {args.save_path}")


if __name__ == "__main__":
    main()
