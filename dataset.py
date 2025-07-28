import os
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class USASegmentationDataset(Dataset):
    def __init__(self, rgb_paths, nir_paths, mask_paths, pil_transform=None, tensor_transform=None):
        self.rgb_paths = rgb_paths
        self.nir_paths = nir_paths
        self.mask_paths = mask_paths
        self.pil_transform = pil_transform
        self.tensor_transform = tensor_transform

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb_img = Image.open(self.rgb_paths[idx]).convert("RGB")
        nir_img = Image.open(self.nir_paths[idx]).convert("L")
        mask_img = Image.open(self.mask_paths[idx]).convert("L")

        # Apply PIL transforms (geometric augmentations) BEFORE tensor conversion
        if self.pil_transform:
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            rgb_img = self.pil_transform(rgb_img)
            torch.manual_seed(seed)
            nir_img = self.pil_transform(nir_img)
            torch.manual_seed(seed)
            mask_img = self.pil_transform(mask_img)

        # Convert to tensors
        rgb_tensor = T.ToTensor()(rgb_img)    # (3, H, W)
        nir_tensor = T.ToTensor()(nir_img)    # (1, H, W)
        mask_tensor = T.ToTensor()(mask_img)  # (1, H, W)
        
        # Combine RGB and NIR channels
        image_4ch = torch.cat([rgb_tensor, nir_tensor], dim=0)  # (4, H, W)

        # Apply tensor transforms (color augmentation) AFTER tensor conversion
        if self.tensor_transform:
            image_4ch = self.tensor_transform(image_4ch)

        # Process mask: squeeze channel dim and convert to binary long tensor
        mask_tensor = mask_tensor.squeeze(0)  # (H, W)
        mask_tensor = (mask_tensor > 0.5).long()

        return image_4ch, mask_tensor



# Example usage:
if __name__ == "__main__":
    # Test the data loading
    loaders = build_loaders("/path/to/your/data", batch_size=4, augment=True)
    
    # Check a batch
    for images, masks in loaders["train"]:
        print(f"Image batch shape: {images.shape}")  # Should be (batch_size, 4, H, W)
        print(f"Mask batch shape: {masks.shape}")    # Should be (batch_size, H, W)
        print(f"Image dtype: {images.dtype}")        # Should be torch.float32
        print(f"Mask dtype: {masks.dtype}")          # Should be torch.int64
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")  # Should be [0, 1]
        print(f"Mask values: {torch.unique(masks)}")  # Should be [0, 1]
        break