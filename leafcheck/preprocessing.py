"""Single deterministic image contract shared by cache, evaluation and website."""

import random
from PIL import Image, ImageOps
import torch

PREPROCESSING_VERSION = "exif_rgb_letterbox_bilinear_v1"


def prepare_pil(source, size):
    source.seek(0)
    image = ImageOps.exif_transpose(source).convert("RGB")
    return ImageOps.pad(image, (size, size), method=Image.Resampling.BILINEAR, color=(128, 128, 128))


def image_bytes(path, size):
    with Image.open(path) as image:
        return prepare_pil(image, size).tobytes()


def prepare_batch(raw, spec, device="cpu", augmentation="none"):
    """Accept NHWC uint8 and produce standardized NCHW float32 tensors."""
    if spec["version"] != PREPROCESSING_VERSION:
        raise ValueError("Unsupported image preprocessing version")
    x = raw.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2).div_(255)
    if augmentation != "none":
        # Reflections and quarter turns preserve whole-image symptom information.
        count = x.shape[0]
        horizontal = (torch.rand(count) < .5).to(device)[:, None, None, None]
        vertical = (torch.rand(count) < .5).to(device)[:, None, None, None]
        x = torch.where(horizontal, x.flip(-1), x)
        x = torch.where(vertical, x.flip(-2), x)
        x = torch.rot90(x, random.randrange(4), (-2, -1))
        if augmentation == "dihedral_color":
            brightness = (.9 + .2 * torch.rand(count)).to(device)[:, None, None, None]
            contrast = (.9 + .2 * torch.rand(count)).to(device)[:, None, None, None]
            centers = x.mean(dim=(-2, -1), keepdim=True)
            x = ((x - centers) * contrast + centers).mul(brightness).clamp_(0, 1)
        elif augmentation != "dihedral":
            raise ValueError(f"Unsupported augmentation: {augmentation}")
    mean = torch.tensor(spec["mean"], device=device)[None, :, None, None]
    std = torch.tensor(spec["std"], device=device)[None, :, None, None]
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)


def image_tensor(path, spec, device="cpu"):
    size = spec["input_size"]
    raw = torch.frombuffer(bytearray(image_bytes(path, size)), dtype=torch.uint8).reshape(1, size, size, 3)
    return prepare_batch(raw, spec, device)
