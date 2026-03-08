from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


class SRBinaryPairDataset(Dataset):
    """
    Loads image pairs (x, x_sr) with binary labels.
    If sr_cache_root is None, x_sr defaults to x (SR-free mode).

    Supported layouts under data_root:
      - real / fake
      - nature / ai
      - 0_real / 1_fake
      - real + multiple fake domain dirs (kandinsky, sd, mj, ...)
    """

    def __init__(self, data_root: str, sr_cache_root: str = None, image_size: int = 256, fake_dirs=None):
        self.data_root = Path(data_root).resolve()
        self.sr_cache_root = Path(sr_cache_root).resolve() if sr_cache_root else None

        self.tf = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
            ]
        )

        real_dir = self.data_root / "real"
        if not real_dir.is_dir():
            real_dir = self.data_root / "nature"
        if not real_dir.is_dir():
            real_dir = self.data_root / "0_real"
        if not real_dir.is_dir():
            raise ValueError(f"Cannot find real/nature/0_real directory under {self.data_root}")

        if fake_dirs is None:
            auto_fake_dirs = []
            fake_dir = self.data_root / "fake"
            if not fake_dir.is_dir():
                fake_dir = self.data_root / "ai"
            if not fake_dir.is_dir():
                fake_dir = self.data_root / "1_fake"
            if fake_dir.is_dir():
                auto_fake_dirs = [fake_dir]
            else:
                for d in sorted(self.data_root.iterdir()):
                    if not d.is_dir():
                        continue
                    if d == real_dir:
                        continue
                    if d.name.startswith("."):
                        continue
                    auto_fake_dirs.append(d)
            fake_dirs = auto_fake_dirs
        else:
            tmp = []
            for d in fake_dirs:
                p = Path(d)
                if not p.is_absolute():
                    p = self.data_root / p
                tmp.append(p.resolve())
            fake_dirs = tmp

        self.samples = []
        for p in sorted(real_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in VALID_EXTS:
                self.samples.append((p, 0))
        for fd in fake_dirs:
            for p in sorted(fd.rglob("*")):
                if p.is_file() and p.suffix.lower() in VALID_EXTS:
                    self.samples.append((p, 1))

        if not self.samples:
            raise ValueError(f"No samples found under {self.data_root}")

    def __len__(self):
        return len(self.samples)

    def _resolve_sr_path(self, x_path: Path):
        if self.sr_cache_root is None:
            return None
        rel = x_path.relative_to(self.data_root)
        candidates = [self.sr_cache_root / rel, self.sr_cache_root / self.data_root.name / rel]
        for cand in candidates:
            if cand.exists():
                return cand
            stem = cand.with_suffix("")
            for ext in VALID_EXTS:
                cand2 = Path(f"{stem}{ext}")
                if cand2.exists():
                    return cand2
        return None

    def __getitem__(self, idx):
        x_path, y = self.samples[idx]
        x = self.tf(Image.open(x_path).convert("RGB"))

        sr_path = self._resolve_sr_path(x_path)
        if sr_path is None:
            x_sr = x
        else:
            x_sr = self.tf(Image.open(sr_path).convert("RGB"))

        return x, x_sr, torch.tensor(y, dtype=torch.float32), str(x_path)
