import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch, sys, os
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np

from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop
import albumentations as A

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


@dataclass
class AugmentationCfg:
    scale: Tuple[float, float] = (0.9, 1.0)
    ratio: Optional[Tuple[float, float]] = None
    color_jitter: Optional[Union[float, Tuple[float, float, float]]] = None
    interpolation: Optional[str] = None
    re_prob: Optional[float] = None
    re_count: Optional[int] = None
    use_timm: bool = False
    augment_dir: Optional[str] = None
    color_image: bool = False


class ResizeMaxSize(nn.Module):

    def __init__(self, max_size, interpolation=InterpolationMode.BICUBIC, fn='max', fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == 'min' else min
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[:2]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = F.resize(img, new_size, self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = F.pad(img, padding=[pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2], fill=self.fill)
        return img


class AlbumentationsTransform2:
    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, img):
        img = np.array(img)  # Convert to numpy array
        img = self.transform(image=img)['image']
        return Image.fromarray(img)  # Convert back to PIL image

# Define the CLAHE transformation
clahe = A.CLAHE(p=1.0, clip_limit=6.0, tile_grid_size=(12, 12))
CLAHE = AlbumentationsTransform2(clahe) # normalizing using Adaptive Histogram Equalization (CLAHE)


def _convert_to_rgb(image):
    return image.convert('RGB')


def image_transform(
        image_size: int,
        is_train: bool,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        resize_longest_max: bool = False,
        color_image: bool = False, # For grayscale images by default
        fill_color: int = 0,
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
        augment_dir: Optional[str] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    if isinstance(aug_cfg, dict):
        aug_cfg = AugmentationCfg(**aug_cfg)
    else:
        aug_cfg = aug_cfg or AugmentationCfg()
    normalize = Normalize(mean=mean, std=std)
    if is_train:
        aug_cfg_dict = {k: v for k, v in asdict(aug_cfg).items() if v is not None}
        use_timm = aug_cfg_dict.pop('use_timm', False)
        augment_dir = aug_cfg_dict.pop('augment_dir', None)
        color_image = aug_cfg_dict.pop('color_image', False)
        if use_timm:
            from timm.data import create_transform  # timm can still be optional
            if isinstance(image_size, (tuple, list)):
                assert len(image_size) >= 2
                input_size = (3,) + image_size[-2:]
            else:
                input_size = (3, image_size, image_size)
            # by default, timm aug randomly alternates bicubic & bilinear for better robustness at inference time
            aug_cfg_dict.setdefault('interpolation', 'random')
            aug_cfg_dict.setdefault('color_jitter', None)  # disable by default
            train_transform = create_transform(
                input_size=input_size,
                is_training=True,
                hflip=0.,
                mean=mean,
                std=std,
                re_mode='pixel',
                **aug_cfg_dict,
            )
        elif augment_dir is not None:
            if not os.path.exists(os.path.join(augment_dir, 'image_augmentation.py')):
                raise ValueError(f'In folder augment_dir ({augment_dir}), there must contain image_augmentation.py')
            sys.path.append(augment_dir)
            from image_augmentation import customized_augmentation
            train_transform = customized_augmentation(image_size, color_image)
        else:
            train_transform = Compose([
                RandomResizedCrop(
                    image_size,
                    scale=aug_cfg_dict.pop('scale'),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ])
            if aug_cfg_dict:
                warnings.warn(f'Unused augmentation cfg items, specify `use_timm` to use ({list(aug_cfg_dict.keys())}).')
        return train_transform
    else: # not training case
        if resize_longest_max:
            transforms = [
                ResizeMaxSize(image_size, fill=fill_color)
            ]
        else:
            transforms = [
                Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
            ]

        transforms.extend([
            _convert_to_rgb,
            CLAHE,
            ToTensor(),
            normalize,
        ])
        return Compose(transforms)