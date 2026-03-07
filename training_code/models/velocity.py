from pathlib import Path
import sys

import torch
import torch.nn as nn


THIS_DIR = Path(__file__).resolve().parent
TRAINING_CODE_DIR = THIS_DIR.parent
if str(TRAINING_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_CODE_DIR))

from models.openaimodel_medical import UNetModel as MedicalUNet  # noqa: E402


def _default_t(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros(x.size(0), 1, device=x.device, dtype=torch.float32)


class StageVelocityUNet(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_hidden: int = 128,
        num_res_blocks: int = 2,
        num_heads: int = 4,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual
        self.unet = MedicalUNet(
            image_size=None,
            in_channels=c_in,
            model_channels=c_hidden,
            out_channels=c_in,
            num_res_blocks=num_res_blocks,
            attention_resolutions=[],
            channel_mult=(1, 1, 2),
            conv_resample=True,
            dims=2,
            dropout=0.0,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=num_heads,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=False,
        )

    def forward(self, x, t=None):
        if t is None:
            t = _default_t(x)
        elif t.ndim == 1:
            t = t.unsqueeze(1)
        t = t.to(device=x.device, dtype=torch.float32)

        out = self.unet(x, t=t)
        if self.residual:
            return out
        return out - x


class RectifierUNet(StageVelocityUNet):
    def __init__(self, c_in: int = 3, c_hidden: int = 128, num_res_blocks: int = 2, num_heads: int = 4):
        super().__init__(
            c_in=c_in,
            c_hidden=c_hidden,
            num_res_blocks=num_res_blocks,
            num_heads=num_heads,
            residual=True,
        )


def discrepancy_from_sr(x_sr: torch.Tensor, x_hat: torch.Tensor):
    delta_map = torch.abs(x_sr - x_hat)
    score = delta_map.mean(dim=[1, 2, 3])
    return delta_map, score

