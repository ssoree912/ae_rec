import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from rectified.sr_binary_pair_dataset import SRBinaryPairDataset  # noqa: E402
from rectified.rectifier_unet import load_rectifier  # noqa: E402
from models.delta_classifier import SmallCNN  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser("Evaluate image-space delta classifier")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--sr_cache_root", type=str, default=None, help="Optional. Omit for SR-free mode.")
    p.add_argument("--rect_ckpt", type=str, required=True)
    p.add_argument("--clf_ckpt", type=str, required=True)

    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--threshold", type=float, default=0.5)

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
    p.add_argument("--concat_input", action="store_true")

    p.add_argument("--result_folder", type=str, default=None)
    p.add_argument("--exp_name", type=str, default="image_delta_eval")
    return p.parse_args()


def load_state_dict_clean(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "clf" in state:
        return state["clf"], state
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state, None


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


def tpr_at_fpr(y_true, y_score, target_fpr=0.05):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return 0.0
    return float(tpr[valid[-1]])


def calculate_acc(y_true, y_pred, thres):
    r_mask = y_true == 0
    f_mask = y_true == 1
    r_acc = accuracy_score(y_true[r_mask], y_pred[r_mask] > thres) if r_mask.any() else float("nan")
    f_acc = accuracy_score(y_true[f_mask], y_pred[f_mask] > thres) if f_mask.any() else float("nan")
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc


@torch.no_grad()
def evaluate_one(clf, rectifier, loader, device, args):
    clf.eval()
    rectifier.eval()

    y_true, y_pred = [], []
    pbar = tqdm(loader, desc="[Eval]", leave=False)
    for x, x_sr, y, _ in pbar:
        x = x.to(device, non_blocking=True)
        x_sr = x_sr.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

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
        logits = clf(clf_in).squeeze(1)
        prob = torch.sigmoid(logits)

        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(prob.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ap = average_precision_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_pred)
    tpr5 = tpr_at_fpr(y_true, y_pred, 0.05)
    r_acc, f_acc, acc = calculate_acc(y_true, y_pred, args.threshold)
    return {
        "ap": float(ap),
        "auroc": float(auroc),
        "tpr@5fpr": float(tpr5),
        "real_acc": float(r_acc),
        "fake_acc": float(f_acc),
        "acc@fixed": float(acc),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    torch.backends.cudnn.benchmark = True

    rectifier = load_rectifier(args.rect_ckpt, device)
    clf_state, meta = load_state_dict_clean(args.clf_ckpt, device)
    if meta is not None and "clf_cfg" in meta:
        c_in = int(meta["clf_cfg"]["c_in"])
        width = int(meta["clf_cfg"]["width"])
    else:
        c_in = 6 if args.concat_input else 3
        width = 64

    clf = SmallCNN(c_in=c_in, width=width).to(device)
    clf.load_state_dict(clf_state, strict=True)
    clf.eval()

    root = Path(args.data_root).resolve()
    real_dir = root / "real"
    if not real_dir.is_dir():
        real_dir = root / "nature"

    fake_dirs = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if d == real_dir:
            continue
        if d.name.startswith("."):
            continue
        fake_dirs.append(d)

    results_rows = []
    y_true_all, y_pred_all = [], []
    for fake_dir in fake_dirs:
        ds = SRBinaryPairDataset(
            data_root=args.data_root,
            sr_cache_root=args.sr_cache_root,
            image_size=args.image_size,
            fake_dirs=[fake_dir],
        )
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )
        out = evaluate_one(clf, rectifier, dl, device, args)

        print(
            f"[{fake_dir.name}] AP={out['ap']:.6f} AUROC={out['auroc']:.6f} "
            f"TPR@5FPR={out['tpr@5fpr']:.6f} ACC@{args.threshold:.2f}={out['acc@fixed']:.6f} "
            f"Real={out['real_acc']:.6f} Fake={out['fake_acc']:.6f}"
        )

        results_rows.append(
            {
                "dataset": fake_dir.name,
                "ap": out["ap"],
                "auroc": out["auroc"],
                "tpr@5fpr": out["tpr@5fpr"],
                "real_acc": out["real_acc"],
                "fake_acc": out["fake_acc"],
                "acc@fixed": out["acc@fixed"],
            }
        )
        y_true_all.append(out["y_true"])
        y_pred_all.append(out["y_pred"])

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    overall = {
        "dataset": "overall",
        "ap": float(average_precision_score(y_true_all, y_pred_all)),
        "auroc": float(roc_auc_score(y_true_all, y_pred_all)),
        "tpr@5fpr": float(tpr_at_fpr(y_true_all, y_pred_all, 0.05)),
        "real_acc": float(accuracy_score(y_true_all[y_true_all == 0], (y_pred_all[y_true_all == 0] > args.threshold))),
        "fake_acc": float(accuracy_score(y_true_all[y_true_all == 1], (y_pred_all[y_true_all == 1] > args.threshold))),
        "acc@fixed": float(accuracy_score(y_true_all, (y_pred_all > args.threshold))),
    }
    results_rows.append(overall)

    print(
        f"[overall] AP={overall['ap']:.6f} AUROC={overall['auroc']:.6f} "
        f"TPR@5FPR={overall['tpr@5fpr']:.6f} ACC@{args.threshold:.2f}={overall['acc@fixed']:.6f}"
    )

    if args.result_folder:
        result_dir = Path(args.result_folder)
        result_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{args.exp_name}"
        with open(result_dir / f"{prefix}_metrics.json", "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "results": results_rows}, f, indent=2)
        pd.DataFrame(results_rows).to_excel(result_dir / f"{prefix}_metrics.xlsx", index=False)
        np.savez(result_dir / f"{prefix}_predictions.npz", y_true=y_true_all, y_pred=y_pred_all)
        print(f"Saved results to {result_dir}")


if __name__ == "__main__":
    main()

