from pathlib import Path
from typing import List, Tuple


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(root: Path):
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )


def discover_class_roots(dataset_root: Path, splits: List[str], classes: List[str]) -> List[Tuple[Path, Path]]:
    """
    Returns list of (input_root, rel_root_from_dataset).
    Supports:
      A) <split>/<class>
      B) <split>/<domain>/<class>
    """
    jobs = []
    class_set = set(classes)

    for split in splits:
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            continue

        # Pattern A
        for cls in classes:
            cls_dir = split_dir / cls
            if cls_dir.is_dir():
                jobs.append((cls_dir, cls_dir.relative_to(dataset_root)))

        # Pattern B
        for child in sorted(split_dir.iterdir()):
            if not child.is_dir():
                continue
            if child.name.startswith("."):
                continue
            if child.name in class_set:
                continue
            for cls in classes:
                cls_dir = child / cls
                if cls_dir.is_dir():
                    jobs.append((cls_dir, cls_dir.relative_to(dataset_root)))

    # Dedup
    out = []
    seen = set()
    for in_root, rel_root in jobs:
        key = str(in_root.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append((in_root, rel_root))
    return out


def resolve_counterpart(root: Path, rel_path: Path):
    direct = root / rel_path
    if direct.exists():
        return direct

    stem = direct.with_suffix("")
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"]:
        cand = Path(f"{stem}{ext}")
        if cand.exists():
            return cand
    return None

