import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import functional as F
from torchvision.transforms import (Normalize, Compose, RandomResizedCrop, RandomRotation, 
                                   RandomHorizontalFlip, ColorJitter, RandomAffine, 
                                   GaussianBlur, RandomCrop, RandomErasing, InterpolationMode, 
                                   ToTensor, Resize, CenterCrop, Lambda)
import random, torch
import numpy as np
from PIL import Image

"""
Example:
Suppose this file is located: '/home/username/Projects/Pytorch-Image-Models/tests/image_augmentation.py'
image_size = (224, 224)

You may import this file as follows:

sys.path.append('/home/username/Projects/Pytorch-Image-Models/tests/image_augmentation.py')
from image_augmentation import customized_augmentation
train_transform = customized_augmentation(image_size)

refs:
https://www.kaggle.com/code/raddar/popular-x-ray-image-normalization-techniques
https://towardsdatascience.com/deep-learning-in-healthcare-x-ray-imaging-part-5-data-augmentation-and-image-normalization-1ead1c02cfe3
"""

class AlbumentationsTransform:
    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, img):
        img = F.to_tensor(img).numpy() * 255  # Convert to numpy array in range [0, 255]
        img = img.transpose(1, 2, 0)  # Convert from CxHxW to HxWxC
        img = self.transform(image=img)['image']
        img = F.to_pil_image(img.astype('uint8'))  # Convert back to PIL image
        return img


class AlbumentationsTransform2:
    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, img):
        img = np.array(img)  # Convert to numpy array
        img = self.transform(image=img)['image']
        return Image.fromarray(img)  # Convert back to PIL image


class RandomAugmentation:
    def __init__(self, probability, *transforms):
        self.probability = probability
        self.transforms = transforms

    def __call__(self, img):
        if random.random() <= self.probability:
            for transform in self.transforms:
                img = transform(img)
        return img

# Define the CLAHE transformation
clahe = A.CLAHE(p=1.0, clip_limit=4.0, tile_grid_size=(8, 8))
CLAHE = AlbumentationsTransform2(clahe), # normalizing using Adaptive Histogram Equalization (CLAHE)

# Define the mean and standard deviation of the dataset
MEAN = (0.3942, 0.3942, 0.3942)#(0.48145466, 0.4578275, 0.40821073)
STD = (0.2378, 0.2378, 0.2378)#(0.26862954, 0.26130258, 0.27577711)
mean_std = Normalize(mean=MEAN, std=STD)

# Define a custom percentile scaling function
def percentile_scaling(img_tensor):
    # Convert tensor to numpy array
    img_array = img_tensor.numpy().squeeze()
    
    # Calculate 2.5th and 97.5th percentiles
    lower = np.percentile(img_array, 2.5)
    upper = np.percentile(img_array, 97.5)
    
    # Clip values outside this range
    img_array = np.clip(img_array, lower, upper)
    img_tensor = (img_tensor - lower) / (upper - lower)
    return img_tensor

def _convert_to_rgb(image):
    return image.convert('RGB')

def customized_augmentation(image_size):
    """
    Customize the augmentation pipeline as per your choice.    
    """
    # Albumentations transformations
    albu_transforms = A.Compose([
        # A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),  # Elastic deformation
        # A.HistogramMatching(p=0.5, reference_images=None),  # Histogram Equalization (note: you need reference images)
        # Add any other albumentations transforms here
        A.GaussianBlur(p=0.2, blur_limit=(3, 5))
    ])
    
    # torchvision transforms
    pre_transforms = Compose([    
        RandomRotation(degrees=15),
        RandomResizedCrop(
            image_size,
            scale=(0.9, 1.1),
            interpolation=InterpolationMode.BICUBIC,
        ),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.15, contrast=0.15),
        RandomAffine(degrees=0, translate=(0.1, 0.1)),
        RandomCrop(size=image_size, padding=10),
        # GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
    ])

    post_transforms = Compose([
        _convert_to_rgb,
        CLAHE, # normalizing using Adaptive Histogram Equalization (CLAHE)
        # Always ensure that ToTensor() is one of the last transformations, because it changes the data type and order of dimensions.
        ToTensor(),
        # mean_std,                      # normalizing using mean + standard deviation
        # Lambda(percentile_scaling),    # normalizing using percentile scaling
        RandomErasing(p=0.2, scale=(0.01, 0.1), ratio=(0.3, 3.3), value=0, inplace=False),  # Cutout or Random Erasing
    ])

    train_transform = Compose([
        # 0.3 means 30% of images will be augmented
        RandomAugmentation(0.3, pre_transforms, AlbumentationsTransform(albu_transforms)),
        post_transforms
    ])

    return train_transform