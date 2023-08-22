import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import functional as F
from torchvision.transforms import (Normalize, Compose, RandomResizedCrop, RandomRotation, 
                                   RandomHorizontalFlip, ColorJitter, RandomAffine, 
                                   GaussianBlur, RandomCrop, RandomErasing, InterpolationMode, 
                                   ToTensor, Resize, CenterCrop)
import random

class AlbumentationsTransform:
    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, img):
        img = F.to_tensor(img).numpy() * 255  # Convert to numpy array in range [0, 255]
        img = img.transpose(1, 2, 0)  # Convert from CxHxW to HxWxC
        img = self.transform(image=img)['image']
        img = F.to_pil_image(img.astype('uint8'))  # Convert back to PIL image
        return img


def _convert_to_rgb(image):
    return image.convert('RGB')

def customized_augmentation(image_size, normalize):
    # Albumentations transformations
    albu_transforms = A.Compose([
        # A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),  # Elastic deformation
        # A.HistogramMatching(p=0.5, reference_images=None),  # Histogram Equalization (note: need reference images)
        # Add any other albumentations transforms here
        A.GaussianBlur(p=0.2, blur_limit=(3, 7))
    ])

    # torchvision transforms
    pre_transforms = Compose([
        RandomRotation(degrees=10),
        RandomResizedCrop(
            image_size,
            scale=(0.9, 1.1),
            interpolation=InterpolationMode.BICUBIC,
        ),
        RandomHorizontalFlip(p=0.3),
        ColorJitter(brightness=0.1, contrast=0.1),
        RandomAffine(degrees=0, translate=(0.1, 0.1)),
        RandomCrop(size=image_size, padding=10),
        # GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
    ])

    post_transforms = Compose([
        _convert_to_rgb,
        ToTensor(),
        normalize,
        RandomErasing(p=0.1, scale=(0.01, 0.10), ratio=(0.3, 3.3), value=0, inplace=False),  # Cutout or Random Erasing
    ])

    class RandomAugmentation:
        def __init__(self, probability, *transforms):
            self.probability = probability
            self.transforms = transforms

        def __call__(self, img):
            if random.random() <= self.probability:
                for transform in self.transforms:
                    img = transform(img)
            return img


    train_transform = Compose([
        # pre_transforms,
        # AlbumentationsTransform(albu_transforms),
        # 0.2 means 20% of images will be augmented
        RandomAugmentation(0.2, pre_transforms, AlbumentationsTransform(albu_transforms)),
        post_transforms
    ])
    """
    The AlbumentationsTransform class acts as a bridge between torchvision and albumentations.
    The HistogramMatching transformation in albumentations requires a set of reference images to match the histogram. You'd need to specify that.
    Always ensure that ToTensor() is one of the last transformations, because it changes the data type and order of dimensions.
    """

    # https://www.kaggle.com/code/raddar/popular-x-ray-image-normalization-techniques
    # https://towardsdatascience.com/deep-learning-in-healthcare-x-ray-imaging-part-5-data-augmentation-and-image-normalization-1ead1c02cfe3

    return train_transform