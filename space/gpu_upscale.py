"""x4 upscale on the ZeroGPU card: the same model as Upscayl's "digital-art-4x" (Real-ESRGAN x4plus anime 6B).

The weights (digital_art_4x.npz, next to this file) were converted 1:1 from Upscayl's ncnn files
digital-art-4x.param / .bin: basicsr RRDBNet with 64 features, 6 blocks, growth 32. Only the upscale runs on the GPU;
tracing stays on the CPU. Pictures are processed in overlapping tiles so very large inputs fit into the card's memory.
"""
import numpy as np
import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

WEIGHTS = Path(__file__).resolve().parent / "digital_art_4x.npz"
NAME = "digital-art-4x (GPU)"


class _RDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class _RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = _RDB(nf, gc), _RDB(nf, gc), _RDB(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, nf=64, nb=23, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
        self.body = nn.Sequential(*[_RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def _load():
    with np.load(WEIGHTS) as npz:
        state = {k: torch.from_numpy(npz[k].astype(np.float32)) for k in npz.files}
    net = RRDBNet(nb=6)
    net.load_state_dict(state, strict=True)
    net.eval()
    return net.to("cuda")              # ZeroGPU: moved to the card at start-up, attached when a GPU call runs


MODEL = _load()


@spaces.GPU(duration=10)
def upscale_x4(bgr, tile=512, pad=16):
    """BGR uint8 (h, w, 3) -> BGR uint8 (4h, 4w, 3)."""
    h, w = bgr.shape[:2]
    rgb = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1])).permute(2, 0, 1).float().div(255)[None].cuda()
    out = torch.empty((1, 3, h * 4, w * 4), dtype=torch.float32, device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                y0, x0 = max(y - pad, 0), max(x - pad, 0)                 # tile plus its overlap
                y1, x1 = min(y + tile + pad, h), min(x + tile + pad, w)
                res = MODEL(rgb[:, :, y0:y1, x0:x1]).float()
                ty1, tx1 = min(y + tile, h), min(x + tile, w)
                out[:, :, y * 4:ty1 * 4, x * 4:tx1 * 4] = res[:, :, (y - y0) * 4:(ty1 - y0) * 4,
                                                              (x - x0) * 4:(tx1 - x0) * 4]
    img = (out[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy())[:, :, ::-1]
    return np.ascontiguousarray(img)


_CPU_MODEL = []


def upscale_x4_cpu(bgr, tile=256, pad=16):
    """The same model on the CPU, used when the GPU quota is used up (slower: minutes instead of seconds)."""
    if not _CPU_MODEL:
        with np.load(WEIGHTS) as npz:
            state = {k: torch.from_numpy(npz[k].astype(np.float32)) for k in npz.files}
        net = RRDBNet(nb=6)
        net.load_state_dict(state, strict=True)
        _CPU_MODEL.append(net.eval())
    model = _CPU_MODEL[0]
    h, w = bgr.shape[:2]
    rgb = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1])).permute(2, 0, 1).float().div(255)[None]
    out = torch.empty((1, 3, h * 4, w * 4), dtype=torch.float32)
    with torch.inference_mode():
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                y0, x0 = max(y - pad, 0), max(x - pad, 0)
                y1, x1 = min(y + tile + pad, h), min(x + tile + pad, w)
                res = model(rgb[:, :, y0:y1, x0:x1])
                ty1, tx1 = min(y + tile, h), min(x + tile, w)
                out[:, :, y * 4:ty1 * 4, x * 4:tx1 * 4] = res[:, :, (y - y0) * 4:(ty1 - y0) * 4,
                                                              (x - x0) * 4:(tx1 - x0) * 4]
    img = (out[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy())[:, :, ::-1]
    return np.ascontiguousarray(img)
