"""View generation for pretraining and segmentation.

The split between CPU and GPU work here is a throughput decision, not a style one.
Kaggle gives 4 vCPUs. At the ~200 img/s a T4 x2 can push through SimCLR, the loader must
produce ~400 augmented 224px views per second; PIL colour-jitter plus Gaussian blur cost
~3 ms/view, which alone caps a 3-worker loader near 1000 views/s and in practice much
lower once JPEG decode is included. So:

    CPU workers:  decode -> RandomResizedCrop -> flip -> uint8 tensor   (cheap)
    GPU, batched: colour jitter -> grayscale -> blur -> solarize -> normalize

I-JEPA and MAE do not need any of this: the released i-jepa configs set
`use_horizontal_flip`, `use_color_distortion` and `use_gaussian_blur` all to False, so
RandomResizedCrop really is the only augmentation. That asymmetry between methods is
intentional and is documented as such in the thesis — equalising augmentation across
four objectives would misrepresent SimCLR and MoCo, whose augmentations *are* the method.
"""
from __future__ import annotations

import random
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ------------------------------------------------------------------ pretraining


class TwoViewTransform:
    """Produce two independently augmented views of one image."""

    def __init__(self, view1: Callable, view2: Callable | None = None) -> None:
        self.view1 = view1
        self.view2 = view2 or view1

    def __call__(self, img):
        return self.view1(img), self.view2(img)


def make_pretrain_transform(
    img_size: int,
    crop_scale: Sequence[float] = (0.3, 1.0),
    horizontal_flip: bool = False,
    to_uint8: bool = False,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
):
    """CPU-side pretraining transform.

    :param to_uint8: return a uint8 CHW tensor instead of a normalised float tensor.
        Used for the contrastive methods, whose photometric augmentation and
        normalisation happen on the GPU in `GPUAugment`. uint8 also makes the
        worker->main-process transfer 4x smaller.
    """
    ops: list = [
        T.RandomResizedCrop(
            img_size, scale=tuple(crop_scale), interpolation=InterpolationMode.BICUBIC
        )
    ]
    if horizontal_flip:
        ops.append(T.RandomHorizontalFlip())
    if to_uint8:
        ops.append(T.PILToTensor())  # uint8 CHW, no scaling
    else:
        ops += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(ops)


class GPUAugment(nn.Module):
    """Batched photometric augmentation on uint8 CUDA tensors.

    Applied per-sample (each image in the batch gets its own random parameters) but
    vectorised where possible. Runs after the batch reaches the GPU, so it costs GPU
    time we have rather than CPU time we do not.
    """

    def __init__(
        self,
        color_jitter_strength: float = 0.4,
        color_distortion: bool = True,
        grayscale_p: float = 0.2,
        jitter_p: float = 0.8,
        gaussian_blur_p: float = 0.5,
        solarize_p: float = 0.0,
        horizontal_flip: bool = True,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        super().__init__()
        s = color_jitter_strength
        self.brightness = 0.8 * s
        self.contrast = 0.8 * s
        self.saturation = 0.8 * s
        self.hue = 0.2 * s
        self.color_distortion = color_distortion
        self.grayscale_p = grayscale_p
        self.jitter_p = jitter_p
        self.blur_p = gaussian_blur_p
        self.solarize_p = solarize_p
        self.hflip = horizontal_flip
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.register_buffer("gray_w", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

    @staticmethod
    def _rand(b: int, lo: float, hi: float, device) -> torch.Tensor:
        return torch.empty(b, 1, 1, 1, device=device).uniform_(lo, hi)

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        """Depthwise Gaussian blur with one shared sigma per batch.

        SimCLR draws sigma per-image; sharing it within a batch keeps this a single
        grouped conv instead of a Python loop, and the sigma still varies every step.
        """
        b, c, h, w = x.shape
        sigma = random.uniform(0.1, 2.0)
        k = max(3, int(0.1 * h) | 1)  # odd kernel, ~10% of image side, as in SimCLR
        coords = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        x = F.conv2d(x, g.view(1, 1, 1, k).expand(c, 1, 1, k), padding=(0, k // 2), groups=c)
        x = F.conv2d(x, g.view(1, 1, k, 1).expand(c, 1, k, 1), padding=(k // 2, 0), groups=c)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """uint8 or float [B,3,H,W] in [0,255] / [0,1] -> normalised float."""
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        b, device = x.shape[0], x.device

        if self.hflip:
            flip = torch.rand(b, device=device) < 0.5
            x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)

        if self.color_distortion:
            apply = (torch.rand(b, 1, 1, 1, device=device) < self.jitter_p).to(x.dtype)
            y = x
            y = y * (1 + apply * (self._rand(b, -self.brightness, self.brightness, device)))
            mean_c = y.mean(dim=(1, 2, 3), keepdim=True)
            y = mean_c + (y - mean_c) * (1 + apply * self._rand(b, -self.contrast, self.contrast, device))
            gray = (y * self.gray_w).sum(dim=1, keepdim=True)
            y = gray + (y - gray) * (1 + apply * self._rand(b, -self.saturation, self.saturation, device))
            # Hue is approximated by a channel-wise shift; a true HSV rotation costs a
            # full colour-space round-trip for a perceptually similar perturbation.
            y = y + apply * self._rand(b, -self.hue, self.hue, device)
            x = y.clamp_(0, 1)

            to_gray = (torch.rand(b, 1, 1, 1, device=device) < self.grayscale_p).to(x.dtype)
            gray = (x * self.gray_w).sum(dim=1, keepdim=True).expand_as(x)
            x = to_gray * gray + (1 - to_gray) * x

        if self.blur_p > 0 and random.random() < self.blur_p:
            x = self._blur(x)

        if self.solarize_p > 0:
            sol = (torch.rand(b, 1, 1, 1, device=device) < self.solarize_p).to(x.dtype)
            x = sol * torch.where(x < 0.5, x, 1.0 - x) + (1 - sol) * x

        return (x - self.mean) / self.std


# ---------------------------------------------------------------- segmentation


class SegTransform:
    """Joint image+mask transform. Every geometric op is applied to both.

    Validation and test use `train=False`, which is *only* resize + normalise. The
    earlier version of this project shared one augmented dataset between train and val
    via a `Subset`, so validation images were randomly flipped — making the val Dice
    noisy and the model-selection signal worse than it looks.
    """

    def __init__(
        self,
        img_size: int,
        train: bool,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
        scale_range: tuple[float, float] = (0.75, 1.25),
        rotation_deg: float = 90.0,
        color_jitter: float = 0.2,
    ) -> None:
        self.img_size = img_size
        self.train = train
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.scale_range = scale_range
        self.rotation_deg = rotation_deg
        self.jitter = T.ColorJitter(color_jitter, color_jitter, color_jitter, color_jitter / 4)

    def __call__(self, img, mask):
        import torchvision.transforms.functional as TF

        if self.train:
            if random.random() < 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() < 0.5:
                img, mask = TF.vflip(img), TF.vflip(mask)
            if self.rotation_deg > 0 and random.random() < 0.5:
                angle = random.uniform(-self.rotation_deg, self.rotation_deg)
                img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
            if self.scale_range and random.random() < 0.5:
                s = random.uniform(*self.scale_range)
                side = max(8, int(self.img_size * s))
                img = TF.resize(img, [side, side], InterpolationMode.BILINEAR)
                mask = TF.resize(mask, [side, side], InterpolationMode.NEAREST)

        img = TF.resize(img, [self.img_size, self.img_size], InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.img_size, self.img_size], InterpolationMode.NEAREST)

        if self.train:
            img = self.jitter(img)

        img = TF.to_tensor(img)
        img = (img - self.mean) / self.std
        mask = TF.to_tensor(mask)
        if mask.shape[0] > 1:  # some Kvasir-SEG masks are saved as RGB
            mask = mask[:1]
        mask = (mask > 0.5).float()
        return img, mask


def make_seg_transforms(img_size: int, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """(train_transform, eval_transform) pair."""
    return (
        SegTransform(img_size, train=True, mean=mean, std=std),
        SegTransform(img_size, train=False, mean=mean, std=std),
    )
