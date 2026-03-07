from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

from .layout import list_images, resolve_counterpart


class RealSRPairDataset(Dataset):
    """
    Real-only rectifier dataset:
      x    : original real image
      x_sr : SR(D(x)) cached image
    """

    def __init__(self, real_root: str, sr_root: str, image_size: int = 256):
        self.real_root = Path(real_root).resolve()
        self.sr_root = Path(sr_root).resolve()
        self.paths = list_images(self.real_root)
        if not self.paths:
            raise ValueError(f"No images found under {self.real_root}")

        self.tf = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        real_path = self.paths[idx]
        rel = real_path.relative_to(self.real_root)
        sr_path = resolve_counterpart(self.sr_root, rel)
        if sr_path is None:
            raise FileNotFoundError(f"Missing SR cache for {real_path} under {self.sr_root}")

        x = self.tf(Image.open(real_path).convert("RGB"))
        x_sr = self.tf(Image.open(sr_path).convert("RGB"))
        return x, x_sr, str(real_path)

