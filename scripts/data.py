import os
import glob
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch

class ChestXRayDataset(torch.utils.data.Dataset):
    """
    Custom dataset for Chest X-Ray segmentation.
    Assumes each image has a corresponding mask with the same file name in a 'masks' folder (all PNG).
    """
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
        
        assert len(self.image_paths) == len(self.mask_paths), "Number of images and masks do not match."
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        if self.transform:
            sample = {"image": image, "mask": mask}
            sample = self.transform(sample)
            image, mask = sample["image"], sample["mask"]
        else:
            image = T.ToTensor()(image)
            mask = T.ToTensor()(mask)
            mask = (mask > 0.5).float()
        return image, mask

class JointTransformWrapper:
    """
    A wrapper to apply transforms that require both image and mask simultaneously (e.g. resize, random flips).
    """
    def __init__(self, augment=True, image_size=(224, 224)):
        self.augment = augment
        self.image_size = image_size
        
        # Common transformation: resize to desired dimensions.
        self.common_transforms = T.Compose([
            T.Resize(self.image_size),
        ])
        
        # Augmentation: random horizontal flip.
        self.augment_transform = T.RandomHorizontalFlip(p=0.5)
        
        # Convert to tensor.
        self.to_tensor_img = T.ToTensor()
        self.to_tensor_mask = T.ToTensor()

        # ImageNet normalization (recommended when using pretrained Swin Transformer).
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        
        # Resize
        image = self.common_transforms(image)
        mask = self.common_transforms(mask)
        
        if self.augment:
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            image = self.augment_transform(image)
            torch.manual_seed(seed)
            mask = self.augment_transform(mask)
        
        # Convert to tensor and normalize image.
        image = self.to_tensor_img(image)
        mask = self.to_tensor_mask(mask)
        image = self.normalize(image)
        
        # Binarize mask.
        mask = (mask > 0.5).float()
        
        return {"image": image, "mask": mask}
