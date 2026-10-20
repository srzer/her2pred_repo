"""Tile encoders and image transforms for GigaPath, ResNet50, and PLIP.

Transforms apply resizing, optional stain normalization and augmentation,
and ImageNet normalization.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# 增广
# ---------------------------------------------------------------------------

class FixedRotation:
    """Rotate by one of {0, 90, 180, 270} degrees, chosen uniformly at random."""
    def __call__(self, img: Image.Image) -> Image.Image:
        angle = random.choice([0, 90, 180, 270])
        return TF.rotate(img, angle)


# ---------------------------------------------------------------------------
# 染色归一化包装成 transform 的一环
# ---------------------------------------------------------------------------

class StainNormTransform:
    """
    把 stain_norm.py 里的 normalizer 包成 PIL → PIL 的可调用对象。

    normalizer 已经按当前切片 fit 过；没 fit 成功的话它内部会退化成恒等映射，
    这里不额外处理，由提取脚本负责告警和计数。
    """

    def __init__(self, normalizer):
        self.normalizer = normalizer

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        return Image.fromarray(self.normalizer.transform(arr))




# ---------------------------------------------------------------------------
# 预处理
# ---------------------------------------------------------------------------

def build_transform(augment: bool, stain_normalizer=None,
                    color_jitter: bool = True,
                    encoder_name: str = 'resnet50') -> transforms.Compose:
    """
    augment=False  →  确定性 resize + normalize（提特征用）
    augment=True   →  随机翻转 / 旋转 /（可选）颜色抖动

    stain_normalizer : stain_norm.py 里的 normalizer 实例，None 表示不做
    color_jitter     : 开染色归一化时建议关掉（见 extract_features.py 的说明）
    """
    # PLIP follows the CLIP image processor, not ImageNet normalization.
    # Keeping this switch here (instead of inside the model) also preserves
    # stain-normalization ordering and lets all encoders share TileDataset.
    if encoder_name == 'plip':
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)

    steps = [transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)]

    if stain_normalizer is not None:
        steps.append(StainNormTransform(stain_normalizer))

    if augment:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            FixedRotation(),
        ]
        if color_jitter:
            steps.append(transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                                saturation=0.05, hue=0.02))

    steps += [transforms.ToTensor(), normalize]
    return transforms.Compose(steps)


# ---------------------------------------------------------------------------
# ResNet50
# ---------------------------------------------------------------------------

class ResNetTileEncoder(nn.Module):
    """
    layer='layer4' : conv1..layer4 → GAP → 2048-d  (默认)
    layer='layer3' : conv1..layer3 → GAP → 1024-d  (CLAM 原文的截断方式)
    """

    OUT_DIM = {'layer3': 1024, 'layer4': 2048}

    def __init__(self, weights: str = 'imagenet_v1', layer: str = 'layer4'):
        super().__init__()
        if layer not in self.OUT_DIM:
            raise ValueError(f"layer must be one of {list(self.OUT_DIM)}, got {layer!r}")

        w = {
            'imagenet_v1': torchvision.models.ResNet50_Weights.IMAGENET1K_V1,
            'imagenet_v2': torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
            'random'     : None,
        }
        if weights not in w:
            raise ValueError(f"weights must be one of {list(w)}, got {weights!r}")

        net = torchvision.models.resnet50(weights=w[weights])

        stages = [net.conv1, net.bn1, net.relu, net.maxpool,
                  net.layer1, net.layer2, net.layer3]
        if layer == 'layer4':
            stages.append(net.layer4)

        self.backbone = nn.Sequential(*stages)
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.out_dim  = self.OUT_DIM[layer]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool(self.backbone(x))
        return torch.flatten(h, 1)


class PLIPTileEncoder(nn.Module):
    """Public pathology CLIP image encoder (vinid/plip), 512-d projection."""

    def __init__(self, model_id: str = 'vinid/plip'):
        super().__init__()
        from transformers import CLIPModel
        self.model = CLIPModel.from_pretrained(model_id)
        self.out_dim = int(self.model.config.projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return the visual projection used by PLIP/CLIP. Do not L2-normalize:
        # CLAM can learn both direction and magnitude, matching other encoders.
        return self.model.get_image_features(pixel_values=x)


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------

def build_encoder(name: str = 'resnet50', device=None,
                  weights: str = 'imagenet_v1', layer: str = 'layer4'):
    """Return an evaluation-mode encoder and its output dimension.

    GigaPath produces 1536-dimensional features, ResNet50 produces 2048 (layer4)
    or 1024 (layer3), and PLIP produces 512. Move the model to device when given.
    """
    if name == 'gigapath':
        import timm
        model   = timm.create_model("hf-hub:prov-gigapath/prov-gigapath",
                                    pretrained=True)
        out_dim = 1536
    elif name == 'resnet50':
        model   = ResNetTileEncoder(weights=weights, layer=layer)
        out_dim = model.out_dim
    elif name == 'plip':
        model = PLIPTileEncoder()
        out_dim = model.out_dim
    else:
        raise ValueError(
            f"unknown encoder {name!r}; choose 'resnet50', 'gigapath', or 'plip'")

    model = model.eval()
    if device is not None:
        model = model.to(device)
    return model, out_dim
