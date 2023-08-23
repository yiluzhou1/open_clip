import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import functional as F
from torchvision.transforms import (Normalize, Compose, RandomResizedCrop, RandomRotation, 
                                   RandomHorizontalFlip, ColorJitter, RandomAffine, 
                                   GaussianBlur, RandomCrop, RandomErasing, InterpolationMode, 
                                   ToTensor, Resize, CenterCrop, Lambda)
import random, torch
import numpy as np
from PIL import Image, ImageOps

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

# If you want to prevent the bottom line of image to be cropped, enter 'bottom'. Otherwise, leave empty string.
crop_preservation = 'bottom' # 'bottom' or ''

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
clahe = A.CLAHE(p=1.0, clip_limit=6.0, tile_grid_size=(12, 12))
CLAHE = AlbumentationsTransform2(clahe) # normalizing using Adaptive Histogram Equalization (CLAHE)

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

def invert_background(img):
    # Convert the background color for xray images
    img = ImageOps.invert(img)
    return img



def customized_augmentation(image_size):
    """
    Customize the augmentation pipeline as per your choice.    
    """
    def resize(img):
        # Resize the image to image_size
        img = img.resize(image_size, Image.BICUBIC)
        return img
    
    def custom_random_resize_crop_preserve_bottom(img):
        """
        TODO: Custom function for Random Crop that ensures the bottom line is preserved.
        It first resizes the image to a randomly chosen size and then crops it to the desired size.
        """
        # Randomly choose a scale between 90% and 100%
        scale_factor = random.uniform(0.9, 1.0)
        
        # Compute the crop dimensions based on the scale
        crop_height = int(img.height * scale_factor)
        crop_width = int(img.width * scale_factor)
        
        # Set the top left corner of the crop
        top = img.height - crop_height
        left = random.randint(0, img.width - crop_width)
        
        img = img.crop((left, top, left + crop_width, img.height))
        resized_img = img.resize(image_size, Image.BICUBIC)
        return resized_img
    
    if crop_preservation == 'bottom':
        # need to preserve the bottom line
        customize_RandomResizedCrop = Compose([
            Lambda(custom_random_resize_crop_preserve_bottom),
            ])
        max_vertical_translate = 0
        
    else:
        customize_RandomResizedCrop = Compose([
        RandomResizedCrop(
            image_size,
            scale=(0.9, 1.1),
            interpolation=InterpolationMode.BICUBIC,
        ),
        ])
        max_vertical_translate = 0.1
    
    # Albumentations transformations
    albu_transforms = A.Compose([
        # A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),  # Elastic deformation
        # A.HistogramMatching(p=0.5, reference_images=None),  # Histogram Equalization (note: you need reference images)
        # Add any other albumentations transforms here
        A.GaussianBlur(p=0.3, blur_limit=(3, 5)),
    ])
    
    # torchvision transforms
    pre_rotation = Compose([
        RandomRotation(degrees=15),
    ])

    pre_resize = Compose([
        Lambda(resize)
    ])

    pre_transforms = Compose([
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.15, contrast=0.15),
        RandomAffine(degrees=0, translate=(0.1, max_vertical_translate)),
        RandomCrop(size=image_size, padding=10), #TODO
        # GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
    ])

    post_transforms = Compose([
        _convert_to_rgb,
        # Always ensure that ToTensor() is one of the last transformations, because it changes the data type and order of dimensions.
        ToTensor(),
        # mean_std,                      # normalizing using mean + standard deviation
        # Lambda(percentile_scaling),    # normalizing using percentile scaling
        RandomErasing(p=0.2, scale=(0.01, 0.1), ratio=(0.3, 3.3), value=0, inplace=False),  # Cutout or Random Erasing
    ])

    train_transform = Compose([
        AlbumentationsTransform2(clahe),
        RandomAugmentation(0.2, Lambda(invert_background)), #20% images: background color will be inverted
        RandomAugmentation(0.3, pre_rotation), # 30% images: RandomRotation
        pre_resize, # all images do resize
        RandomAugmentation(0.25, customize_RandomResizedCrop), # 25% images will have be resized here
        # 0.3 means 30% of images will be augmented
        RandomAugmentation(0.3, pre_transforms, AlbumentationsTransform(albu_transforms)),
        post_transforms
    ])

    return train_transform