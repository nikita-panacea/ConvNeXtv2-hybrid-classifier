# utils/augment.py
"""
Data augmentation utilities for skin lesion classification.

Paper mentions the following augmentation techniques:
- Rotation
- Flipping (horizontal and vertical)
- Scaling
- Smoothing
- Mix-up
- Color jitter

ImageNet normalization is used since we're using ImageNet pretrained weights.
"""

from torchvision import transforms
from torchvision.transforms import functional as TF
import torch
import numpy as np
from PIL import ImageFilter
import random


# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class GaussianSmooth:
    """Apply Gaussian smoothing with random radius."""
    def __init__(self, radius_range=(0.5, 1.5)):
        self.radius_range = radius_range
    
    def __call__(self, img):
        if random.random() < 0.3:  # Apply with 30% probability
            radius = random.uniform(*self.radius_range)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


def build_transforms(train=True, input_size=224, aa='rand-m9-mstd0.5-inc1',
                     reprob=0.0, color_jitter=0.4, use_mixup=False):
    """
    Build data transforms for training or validation.
    
    Args:
        train: Whether to build training transforms (with augmentation)
        input_size: Target image size (paper uses 224)
        aa: AutoAugment policy (not currently used, for future)
        reprob: Random erasing probability
        color_jitter: Color jitter strength
        use_mixup: Whether to use mixup (handled in training loop)
    
    Returns:
        torchvision.transforms.Compose object
    """
    if train:
        # Training transforms with augmentation as per paper
        train_transforms = [
            # Scaling (random resize crop)
            transforms.RandomResizedCrop(
                input_size, 
                scale=(0.6, 1.0),  # Random scaling between 60% and 100%
                ratio=(0.75, 1.33),  # Aspect ratio range
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            
            # Flipping (horizontal and vertical)
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            
            # Rotation (random rotation up to 20 degrees)
            transforms.RandomRotation(
                degrees=20,
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            
            # Smoothing (Gaussian blur with small probability)
            GaussianSmooth(radius_range=(0.5, 1.5)),
            
            # Color jitter
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=color_jitter * 0.25  # Smaller hue variation
            ),
            
            # Random erasing (additional augmentation for robustness)
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]

        # Optional random erasing (not mentioned in paper; default off)
        if reprob and reprob > 0:
            train_transforms.append(
                transforms.RandomErasing(
                    p=reprob,
                    scale=(0.02, 0.33),
                    ratio=(0.3, 3.3),
                    value='random'
                )
            )

        return transforms.Compose(train_transforms)
    else:
        # Validation/test transforms (no augmentation)
        return transforms.Compose([
            # Resize to slightly larger than target, then center crop
            # This ensures consistent evaluation across different image sizes
            transforms.Resize(
                int(input_size / 0.875),  # ~256 for input_size=224
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


class MixupAugmentation:
    """
    Mixup augmentation for training.
    
    Paper mentions mix-up as one of the augmentation techniques.
    This should be applied in the training loop, not in the transform pipeline.
    
    Reference: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2017)
    
    Args:
        alpha: Mixup interpolation coefficient (default: 0.8)
    """
    def __init__(self, alpha=0.8):
        self.alpha = alpha
    
    def __call__(self, x, y):
        """
        Apply mixup to a batch.
        
        Args:
            x: Input batch tensor [B, C, H, W]
            y: Label batch tensor [B]
        
        Returns:
            mixed_x: Mixed input batch
            y_a: Original labels
            y_b: Shuffled labels
            lam: Mixing coefficient
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute mixup loss.
    
    Args:
        criterion: Loss function (e.g., CrossEntropyLoss)
        pred: Model predictions
        y_a: Original labels
        y_b: Shuffled labels
        lam: Mixing coefficient
    
    Returns:
        Mixed loss value
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Additional augmentation transforms for experiments
def build_strong_augmentation(input_size=224):
    """
    Build stronger augmentation pipeline for experiments.
    
    This includes more aggressive transformations that may help
    with the challenging ISIC 2019 dataset.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            input_size,
            scale=(0.5, 1.0),
            ratio=(0.75, 1.33),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=10,
        ),
        GaussianSmooth(radius_range=(0.5, 2.0)),
        transforms.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.1,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.4)),
    ])


def get_test_time_augmentation_transforms(input_size=224):
    """
    Get transforms for test-time augmentation (TTA).
    
    Returns a list of transforms that can be applied to create
    multiple augmented versions of the same image for ensemble prediction.
    """
    base_transform = transforms.Compose([
        transforms.Resize(int(input_size / 0.875)),
        transforms.CenterCrop(input_size),
    ])
    
    final_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    # TTA variants
    tta_transforms = [
        # Original
        transforms.Compose([base_transform, final_transform]),
        # Horizontal flip
        transforms.Compose([base_transform, transforms.RandomHorizontalFlip(p=1.0), final_transform]),
        # Vertical flip
        transforms.Compose([base_transform, transforms.RandomVerticalFlip(p=1.0), final_transform]),
        # Both flips
        transforms.Compose([base_transform, transforms.RandomHorizontalFlip(p=1.0), 
                           transforms.RandomVerticalFlip(p=1.0), final_transform]),
        # 90 degree rotation
        transforms.Compose([
            base_transform,
            transforms.Lambda(lambda img: TF.rotate(img, 90)),
            final_transform,
        ]),
    ]
    
    return tta_transforms
