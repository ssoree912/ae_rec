import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class RectifierUNet(nn.Module):
    """
    Shallow residual U-Net:
      input  : SR(x)
      output : x_hat = SR(x) + f(SR(x))
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.enc1 = ConvBlock(in_channels, c1)
        self.down1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.enc2 = ConvBlock(c2, c2)
        self.down2 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)
        self.bottleneck = ConvBlock(c3, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 + c1, c1)

        self.out = nn.Conv2d(c1, in_channels, kernel_size=1)

    def forward(self, x_sr: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x_sr)
        e2 = self.enc2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))

        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        residual = self.out(d1)
        return (x_sr + residual).clamp(0.0, 1.0)


def load_rectifier(ckpt_path: str, device: torch.device) -> RectifierUNet:
    data = torch.load(ckpt_path, map_location=device)
    model_cfg = data.get("model_cfg", {})
    model = RectifierUNet(
        in_channels=int(model_cfg.get("in_channels", 3)),
        base_channels=int(model_cfg.get("base_channels", 32)),
    )
    state = data["model"] if "model" in data else data
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model

