import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

from rectified.rectifier_unet import load_rectifier


def parse_args():
    p = argparse.ArgumentParser(description="Compute rectified discrepancy score for robustness validation")
    p.add_argument("--sr_input", type=str, required=True, help="Folder containing SR images")
    p.add_argument("--rectifier_ckpt", type=str, required=True)
    p.add_argument("--csv_output", type=str, required=True)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def list_images(root: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def main():
    args = parse_args()
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    rectifier = load_rectifier(args.rectifier_ckpt, device=device)
    rectifier.eval()

    tf = T.Compose([T.Resize((args.image_size, args.image_size)), T.ToTensor()])
    paths = list_images(Path(args.sr_input).resolve())
    if len(paths) == 0:
        raise ValueError(f"No images found in {args.sr_input}")

    Path(args.csv_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv_output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "delta_score"])

        for p in tqdm(paths, desc="validate_for_robustness"):
            x_sr = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                x_hat = rectifier(x_sr)
                score = torch.abs(x_sr - x_hat).mean().item()
            writer.writerow([str(p), score])

    print(f"[Done] wrote {len(paths)} rows -> {args.csv_output}")


if __name__ == "__main__":
    main()

