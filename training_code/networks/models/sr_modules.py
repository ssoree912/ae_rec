import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F


# torchvision / basicsr compatibility:
# Some basicsr versions import `torchvision.transforms.functional_tensor`,
# which is removed in newer torchvision releases.
try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ModuleNotFoundError:
    from torchvision.transforms.functional import rgb_to_grayscale

    functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
    functional_tensor.rgb_to_grayscale = rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _first_existing_path(candidates):
    for p in candidates:
        if not p:
            continue
        p = os.path.abspath(os.path.expanduser(p))
        if os.path.isdir(p):
            return p
    return None


def _resolve_realesrgan_root():
    env_root = os.getenv("REALESRGAN_ROOT")
    candidates = [
        env_root,
        os.path.join(PROJECT_ROOT, "Real-ESRGAN"),
        os.path.join(os.path.dirname(PROJECT_ROOT), "D3", "Real-ESRGAN"),
    ]
    resolved = _first_existing_path(candidates)
    if resolved is None:
        resolved = os.path.join(PROJECT_ROOT, "Real-ESRGAN")
    return resolved


REALESRGAN_ROOT = _resolve_realesrgan_root()
if REALESRGAN_ROOT not in sys.path:
    sys.path.insert(0, REALESRGAN_ROOT)

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Failed to import Real-ESRGAN modules. "
        f"Expected local path: {REALESRGAN_ROOT}. "
        "Install deps with `pip install -r Real-ESRGAN/requirements.txt` "
        "or `pip install -e Real-ESRGAN`."
    ) from e


def _resolve_model_path(model_name: str):
    direct = os.path.abspath(os.path.expanduser(model_name))
    if os.path.isfile(direct):
        return direct

    filename = model_name if model_name.endswith(".pth") else f"{model_name}.pth"
    candidates = []

    env_weights = os.getenv("REALESRGAN_WEIGHTS_DIR")
    if env_weights:
        candidates.append(os.path.join(os.path.abspath(os.path.expanduser(env_weights)), filename))

    candidates.extend(
        [
            os.path.join(REALESRGAN_ROOT, "weights", filename),
            os.path.join(REALESRGAN_ROOT, "experiments", "pretrained_models", filename),
            os.path.join(os.path.dirname(PROJECT_ROOT), "D3", "Real-ESRGAN", "weights", filename),
            os.path.join(PROJECT_ROOT, "weights", filename),
        ]
    )

    for p in candidates:
        if os.path.isfile(p):
            return p

    searched = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"Model file not found for model_name='{model_name}'. Searched:\n{searched}\n"
        "Set REALESRGAN_WEIGHTS_DIR or pass a direct .pth path via --sr_model_name."
    )


class BasicSRProcessor:
    """Downsample + Real-ESRGAN upsample processor."""

    def __init__(self, scale=4, model_name="RealESRGAN_x4plus", device="cuda", tile=512):
        self.scale = scale
        self.device = device
        use_half = str(device).startswith("cuda")

        model_path = _resolve_model_path(model_name)
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale,
        )

        self.upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=use_half,
            device=device,
        )

        print(
            f"[SR] initialized model={model_name} "
            f"scale={scale} root={REALESRGAN_ROOT}"
        )

    def downsample(self, img_tensor):
        return F.interpolate(
            img_tensor,
            scale_factor=1 / self.scale,
            mode="bicubic",
            align_corners=False,
        )

    def tensor_to_numpy(self, tensor):
        numpy_imgs = []
        for i in range(tensor.size(0)):
            img = tensor[i].permute(1, 2, 0).detach().cpu().numpy()
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
            numpy_imgs.append(img)
        return numpy_imgs

    def numpy_to_tensor(self, numpy_imgs):
        tensors = []
        for img in numpy_imgs:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            tensors.append(img_tensor)
        return torch.stack(tensors).to(self.device)

    def sr_process(self, img_tensor):
        downsampled = self.downsample(img_tensor)
        numpy_imgs = self.tensor_to_numpy(downsampled)

        sr_numpy_imgs = []
        for img in numpy_imgs:
            sr_img, _ = self.upsampler.enhance(img, outscale=self.scale)
            sr_numpy_imgs.append(sr_img)

        return self.numpy_to_tensor(sr_numpy_imgs)

    def process_batch(self, batch_tensor):
        return self.sr_process(batch_tensor)


def get_sr_processor(scale=4, model_name="RealESRGAN_x4plus", device="cuda", tile=512):
    return BasicSRProcessor(
        scale=scale,
        model_name=model_name,
        device=device,
        tile=tile,
    )

