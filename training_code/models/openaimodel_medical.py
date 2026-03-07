from pathlib import Path
import sys

import torch.nn as nn


THIS_DIR = Path(__file__).resolve().parent
TRAINING_CODE_DIR = THIS_DIR.parent
if str(TRAINING_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_CODE_DIR))

from rectified.rectifier_unet import RectifierUNet as _RectifierUNet  # noqa: E402


class UNetModel(nn.Module):
    """
    Lightweight compatibility adapter.
    Keeps the D3-style class name while delegating to the local rectifier U-Net.
    """

    def __init__(
        self,
        image_size=None,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=None,
        channel_mult=(1, 1, 2),
        conv_resample=True,
        dims=2,
        dropout=0.0,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=False,
    ):
        super().__init__()
        _ = (
            image_size,
            num_res_blocks,
            attention_resolutions,
            channel_mult,
            conv_resample,
            dims,
            dropout,
            num_classes,
            use_checkpoint,
            use_fp16,
            num_heads,
            num_head_channels,
            num_heads_upsample,
            use_scale_shift_norm,
            resblock_updown,
        )
        base_channels = max(16, int(model_channels) // 4)
        self.net = _RectifierUNet(in_channels=in_channels, base_channels=base_channels)

    def forward(self, x, t=None):
        _ = t
        y = self.net(x)
        if y.shape[1] != x.shape[1]:
            return y[:, : x.shape[1]]
        return y

