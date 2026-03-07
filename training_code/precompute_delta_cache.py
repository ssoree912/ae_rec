import argparse
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm

from rectified.layout import discover_class_roots, list_images, resolve_counterpart
from rectified.rectifier_unet import load_rectifier


def parse_args():
    p = argparse.ArgumentParser(description="Precompute delta = |SR(x) - R(SR(x))| cache")
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--sr_root", type=str, required=True)
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--rectifier_ckpt", type=str, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "valid"])
    p.add_argument("--classes", nargs="+", default=["0_real", "1_fake"])
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_ext", type=str, default="keep", choices=["keep", "png", "jpg"])
    p.add_argument("--jpg_quality", type=int, default=95)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def build_output_path(src_path: Path, input_root: Path, output_root: Path, save_ext: str):
    rel = src_path.relative_to(input_root)
    if save_ext == "keep":
        out_rel = rel
    elif save_ext == "png":
        out_rel = rel.with_suffix(".png")
    else:
        out_rel = rel.with_suffix(".jpg")
    return output_root / out_rel


class SRJobDataset(Dataset):
    def __init__(self, input_root: Path, sr_root: Path, output_root: Path, image_size: int, save_ext: str, max_items=None):
        self.input_root = input_root
        self.sr_root = sr_root
        self.output_root = output_root
        self.save_ext = save_ext
        self.tf = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])

        images = list_images(input_root)
        if max_items is not None:
            images = images[:max_items]

        self.items = []
        for src_path in images:
            rel = src_path.relative_to(input_root)
            sr_path = resolve_counterpart(sr_root, rel)
            if sr_path is None:
                continue
            out_path = build_output_path(src_path, input_root, output_root, save_ext)
            self.items.append((sr_path, out_path))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sr_path, out_path = self.items[idx]
        x_sr = self.tf(Image.open(sr_path).convert("RGB"))
        return x_sr, str(out_path)


def save_delta_batch(delta, out_paths, save_ext, jpg_quality):
    delta = delta.detach().cpu().clamp(0, 1)
    for i, out in enumerate(out_paths):
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        arr = (delta[i].permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
        img = Image.fromarray(arr)
        if save_ext == "jpg" or (save_ext == "keep" and out_path.suffix.lower() in {".jpg", ".jpeg"}):
            img.save(out_path, quality=jpg_quality)
        else:
            img.save(out_path)


def run_job(input_root, sr_root, out_root, rectifier, args, device):
    ds = SRJobDataset(
        input_root=input_root,
        sr_root=sr_root,
        output_root=out_root,
        image_size=args.image_size,
        save_ext=args.save_ext,
        max_items=args.max_items,
    )
    if len(ds) == 0:
        print(f"[SKIP] no matched SR files for {input_root}")
        return

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"[Delta] input={input_root} sr={sr_root} out={out_root} count={len(ds)}")
    seen = 0
    for x_sr, out_paths in tqdm(loader, desc=str(input_root), leave=False):
        to_save = []
        to_save_paths = []
        for j, out in enumerate(out_paths):
            if (not args.overwrite) and Path(out).exists():
                continue
            to_save.append(x_sr[j])
            to_save_paths.append(out)

        if len(to_save) == 0:
            seen += len(out_paths)
            if seen % args.log_every == 0:
                print(f"[{seen}/{len(ds)}] all skipped (exists)")
            continue

        batch = torch.stack(to_save, dim=0).to(device, non_blocking=True)
        with torch.no_grad():
            x_hat = rectifier(batch)
            delta = torch.abs(batch - x_hat)
        save_delta_batch(delta, to_save_paths, args.save_ext, args.jpg_quality)

        seen += len(out_paths)
        if seen % args.log_every == 0:
            print(f"[{seen}/{len(ds)}] processed")


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    sr_root = Path(args.sr_root).resolve()
    output_root = Path(args.output_root).resolve()
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    jobs = discover_class_roots(dataset_root, args.splits, args.classes)
    if len(jobs) == 0:
        raise ValueError("No class roots found. Check --dataset_root/--splits/--classes")

    print(f"[Delta] jobs={len(jobs)}")
    for i, (input_root, rel_root) in enumerate(jobs, start=1):
        print(f"[{i}] {input_root} -> {output_root / rel_root}")

    if args.dry_run:
        return

    rectifier = load_rectifier(args.rectifier_ckpt, device=device)
    rectifier.eval()

    for input_root, rel_root in jobs:
        run_job(
            input_root=input_root,
            sr_root=sr_root / rel_root,
            out_root=output_root / rel_root,
            rectifier=rectifier,
            args=args,
            device=device,
        )


if __name__ == "__main__":
    main()

