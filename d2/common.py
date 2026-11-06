"""Shared definitions for D2 qualification training and inference."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
from PIL import Image, ImageOps

# Order of the four sites, matching the training CSV columns.
SITES = [
    ("upper_extremity", "left", "arm_left"),
    ("upper_extremity", "right", "arm_right"),
    ("lower_extremity", "left", "leg_left"),
    ("lower_extremity", "right", "leg_right"),
]
CLASSES = ["no_injury", "wound", "amputation", "not_testable"]
# Mirroring an image swaps the casualty's anatomical left and right.
FLIP_PERM = [1, 0, 3, 2]

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

try:  # Some dataset ".png" files are really HEIC; decode by content.
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover - optional dependency
    pass


def load_image(path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


class SiteClassifier(nn.Module):
    """Backbone + one 4-way head per body site."""

    def __init__(self, backbone: str, pretrained_path: str | None = None, drop: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")
            state = {k: v for k, v in state.items() if not k.startswith("classifier.")}
            self.backbone.load_state_dict(state, strict=False)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(self.backbone.num_features, len(SITES) * len(CLASSES))

    def forward(self, x):
        f = self.drop(self.backbone(x))
        return self.head(f).view(-1, len(SITES), len(CLASSES))


class _FineTuned(nn.Module):
    def __init__(self, model: SiteClassifier, size: int):
        super().__init__()
        self.model, self.size = model, size

    def predict_proba(self, x):
        return self.model(x).softmax(-1)


def load_model(path, device="cpu"):
    """Load a fine-tuned SiteClassifier checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m = SiteClassifier(ckpt["backbone"])
    m.load_state_dict(ckpt["state_dict"])
    return _FineTuned(m, ckpt["size"]).to(device).eval()
