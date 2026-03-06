import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from networks.models.sr_modules import BasicSRProcessor


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_SPLITS = ("train", "val", "test")
DEFAULT_CLASSES = ("0_real", "1_fake", "real", "fake", "nature", "ai")
DEFAULT_REAL_NAMES = ("0_real", "real", "nature")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute SR(D(x)) cache with dataset-layout presets."
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="recursive",
        choices=["recursive", "images_original", "split_class"],
        help=(
            "recursive: single input_root->output_root recursive copy; "
            "images_original: dataset_root/images/original/<domain>; "
            "split_class: dataset_root/<split>/(<class> or <domain>/<class>)."
        ),
    )
    parser.add_argument("--dataset_root", type=str, default=None, help="Root for layout presets")
    parser.add_argument("--input_root", type=str, default=None, help="Input root (recursive mode)")
    parser.add_argument("--output_root", type=str, default=None, help="Output root/base")
    parser.add_argument(
        "--extra_job",
        nargs=2,
        action="append",
        default=[],
        metavar=("INPUT_ROOT", "OUTPUT_ROOT"),
        help="Optional extra jobs. Can be repeated.",
    )
    parser.add_argument(
        "--source_subdir",
        type=str,
        default="images/original",
        help="Used in images_original layout: source subdir under dataset_root",
    )
    parser.add_argument(
        "--target_subdir",
        type=str,
        default="images/post-processed",
        help="Used in images_original layout: default output subdir under dataset_root",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Used in split_class layout, e.g. train val test",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_CLASSES),
        help="Used in split_class layout, e.g. 0_real 1_fake or real fake",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Optional domain folder names to include in layout presets",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sr_model_name", type=str, default="RealESRGAN_x4plus")
    parser.add_argument("--sr_scale", type=int, default=4)
    parser.add_argument("--sr_tile", type=int, default=512)
    parser.add_argument("--save_ext", type=str, default="keep", choices=["keep", "png", "jpg"])
    parser.add_argument("--jpg_quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--only_real", action="store_true", help="Process real-class images only")
    parser.add_argument(
        "--real_names",
        nargs="+",
        default=list(DEFAULT_REAL_NAMES),
        help="Folder names regarded as real class when --only_real is used",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_empty", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device_arg


def list_images(root: Path):
    paths = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            paths.append(p)
    return sorted(paths)


def get_real_name_set(args):
    return {x.strip().lower() for x in args.real_names if x and x.strip()}


def unique_jobs(jobs):
    out = []
    seen = set()
    for in_root, out_root in jobs:
        key = (str(in_root.resolve()), str(out_root.resolve()))
        if key in seen:
            continue
        seen.add(key)
        out.append((in_root, out_root))
    return out


def build_jobs_recursive(args):
    if not args.input_root or not args.output_root:
        raise ValueError("--layout recursive requires both --input_root and --output_root.")
    return [(Path(args.input_root).resolve(), Path(args.output_root).resolve())]


def build_jobs_images_original(args):
    if not args.dataset_root:
        raise ValueError("--layout images_original requires --dataset_root.")
    dataset_root = Path(args.dataset_root).resolve()
    src_base = dataset_root / args.source_subdir
    out_base = Path(args.output_root).resolve() if args.output_root else (dataset_root / args.target_subdir)

    if not src_base.is_dir():
        raise FileNotFoundError(f"Source folder not found: {src_base}")

    domain_filter = set(args.domains) if args.domains else None
    real_names = get_real_name_set(args)
    jobs = []
    for d in sorted(src_base.iterdir()):
        if not d.is_dir():
            continue
        if args.only_real and d.name.lower() not in real_names:
            continue
        if domain_filter is not None and d.name not in domain_filter:
            continue
        jobs.append((d, out_base / d.name))
    if not jobs:
        raise ValueError(f"No domain folders found under: {src_base}")
    return jobs


def build_jobs_split_class(args):
    if not args.dataset_root:
        raise ValueError("--layout split_class requires --dataset_root.")
    dataset_root = Path(args.dataset_root).resolve()
    src_base = Path(args.input_root).resolve() if args.input_root else dataset_root
    out_base = Path(args.output_root).resolve() if args.output_root else (dataset_root / "sr_cache")

    real_names = get_real_name_set(args)
    classes = list(args.classes)
    if args.only_real:
        classes = [c for c in classes if c.lower() in real_names]
        if not classes:
            raise ValueError(
                "No real classes left after --only_real filtering. "
                f"classes={args.classes}, real_names={args.real_names}"
            )

    class_names = set(classes)
    domain_filter = set(args.domains) if args.domains else None
    jobs = []
    for split in args.splits:
        split_dir = src_base / split
        if not split_dir.is_dir():
            continue

        # Pattern A: <split>/<class>
        for cls in classes:
            cls_dir = split_dir / cls
            if cls_dir.is_dir():
                jobs.append((cls_dir, out_base / split / cls))

        # Pattern B: <split>/<domain>/<class>
        for domain_dir in sorted(split_dir.iterdir()):
            if not domain_dir.is_dir():
                continue
            if domain_dir.name.startswith("."):
                continue
            if domain_dir.name in class_names:
                continue
            if domain_filter is not None and domain_dir.name not in domain_filter:
                continue
            for cls in classes:
                cls_dir = domain_dir / cls
                if cls_dir.is_dir():
                    jobs.append((cls_dir, out_base / split / domain_dir.name / cls))
    if not jobs:
        raise ValueError(
            f"No class folders found under {src_base}. "
            f"Checked splits={args.splits}, classes={args.classes}."
        )
    return jobs


def build_jobs(args):
    if args.layout == "recursive":
        jobs = build_jobs_recursive(args)
    elif args.layout == "images_original":
        jobs = build_jobs_images_original(args)
    else:
        jobs = build_jobs_split_class(args)

    for in_root, out_root in args.extra_job:
        jobs.append((Path(in_root).resolve(), Path(out_root).resolve()))

    jobs = unique_jobs(jobs)
    if not jobs:
        raise ValueError("No jobs to run.")
    return jobs


def build_output_path(src_path: Path, input_root: Path, output_root: Path, save_ext: str):
    rel = src_path.relative_to(input_root)
    if save_ext == "keep":
        out_rel = rel
    elif save_ext == "png":
        out_rel = rel.with_suffix(".png")
    else:
        out_rel = rel.with_suffix(".jpg")
    return output_root / out_rel


def tensor_to_uint8_image(t: torch.Tensor):
    t = t.detach().cpu().clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return arr


def is_real_path(src_path: Path, input_root: Path, real_names):
    # If the input root itself is a real folder, accept all files beneath it.
    if input_root.name.lower() in real_names:
        return True
    rel = src_path.relative_to(input_root)
    dir_parts = [x.lower() for x in rel.parts[:-1]]
    return any(part in real_names for part in dir_parts)


def run_job(input_root: Path, output_root: Path, sr: BasicSRProcessor, device: str, args):
    if not input_root.exists():
        raise FileNotFoundError(f"input_root does not exist: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    paths = list_images(input_root)
    if args.only_real:
        real_names = get_real_name_set(args)
        paths = [p for p in paths if is_real_path(p, input_root, real_names)]
    if args.max_items is not None:
        paths = paths[: args.max_items]
    if not paths:
        raise ValueError(f"No images found under: {input_root}")

    total = len(paths)
    saved = 0
    skipped = 0
    failed = 0

    print(f"[SR Cache] input_root={input_root}")
    print(f"[SR Cache] output_root={output_root}")
    print(f"[SR Cache] total_images={total}")

    for i, src_path in enumerate(paths, start=1):
        out_path = build_output_path(src_path, input_root, output_root, args.save_ext)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not args.overwrite:
            skipped += 1
            if i % args.log_every == 0:
                print(f"[{i}/{total}] skipped={skipped} saved={saved} failed={failed}")
            continue

        try:
            img = Image.open(src_path).convert("RGB")
            x = TF.to_tensor(img).unsqueeze(0).to(device)
            with torch.no_grad():
                x_sr = sr.sr_process(x)[0]

            sr_np = tensor_to_uint8_image(x_sr)
            sr_img = Image.fromarray(sr_np)

            if args.save_ext == "jpg":
                sr_img.save(out_path, quality=args.jpg_quality)
            elif args.save_ext == "keep" and out_path.suffix.lower() in {".jpg", ".jpeg"}:
                sr_img.save(out_path, quality=args.jpg_quality)
            else:
                sr_img.save(out_path)
            saved += 1
        except Exception as e:
            failed += 1
            print(f"[ERROR] {src_path}: {e}")

        if i % args.log_every == 0:
            print(f"[{i}/{total}] skipped={skipped} saved={saved} failed={failed}")

    print(f"[DONE] total={total} saved={saved} skipped={skipped} failed={failed}")


def main():
    args = parse_args()
    device = resolve_device(args.device)
    jobs = build_jobs(args)

    print(f"[SR Cache] layout={args.layout} jobs={len(jobs)}")
    for idx, (in_root, out_root) in enumerate(jobs, start=1):
        print(f"[{idx}] {in_root} -> {out_root}")

    if args.dry_run:
        return

    sr = BasicSRProcessor(
        scale=args.sr_scale,
        model_name=args.sr_model_name,
        device=device,
        tile=args.sr_tile,
    )

    for idx, (input_root, output_root) in enumerate(jobs, start=1):
        print(f"\n=== Job {idx}/{len(jobs)} ===")
        try:
            run_job(input_root=input_root, output_root=output_root, sr=sr, device=device, args=args)
        except ValueError as e:
            if args.skip_empty and "No images found under" in str(e):
                print(f"[SKIP_EMPTY] {e}")
                continue
            raise


if __name__ == "__main__":
    main()
