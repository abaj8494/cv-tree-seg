import os
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from unet import UNet 


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

        # Apply tensor transforms (ONLY to image, NOT mask)
        if self.tensor_transform:
            image_4ch = self.tensor_transform(image_4ch)

        # Process mask: squeeze channel dim and convert to binary long tensor
        mask_tensor = mask_tensor.squeeze(0)  # (H, W)
        mask_tensor = (mask_tensor > 0.5).long()

        return image_4ch, mask_tensor


def get_image_pairs(root_dir):
    rgb_dir = os.path.join(root_dir, "RGB_images")
    nir_dir = os.path.join(root_dir, "NRG_images")
    mask_dir = os.path.join(root_dir, "masks")

    rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    nir_paths = sorted(glob.glob(os.path.join(nir_dir, "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    assert len(rgb_paths) == len(nir_paths) == len(mask_paths), "Mismatch in file counts"

    return rgb_paths, nir_paths, mask_paths


def analyze_dataset(root_dir):
    """Analyze the dataset to understand class distribution"""
    _, _, mask_paths = get_image_pairs(root_dir)
    
    total_pixels = 0
    dead_tree_pixels = 0
    
    print("Analyzing dataset...")
    for mask_path in tqdm(mask_paths[:50]):  # Sample first 50 images
        mask = Image.open(mask_path).convert("L")
        mask_array = np.array(mask)
        
        total_pixels += mask_array.size
        dead_tree_pixels += np.sum(mask_array > 128)  # White pixels
    
    dead_ratio = dead_tree_pixels / total_pixels
    print(f"Dead tree pixel ratio: {dead_ratio:.4f}")
    print(f"Background pixel ratio: {1-dead_ratio:.4f}")
    
    return dead_ratio


def build_loaders(root_dir, batch_size=8, val_ratio=0.1, test_ratio=0.2, augment=False):
    rgb_paths, nir_paths, mask_paths = get_image_pairs(root_dir)

    # First split: separate test set
    temp_data = list(zip(rgb_paths, nir_paths, mask_paths))
    train_val_data, test_data = train_test_split(
        temp_data, test_size=test_ratio, random_state=42
    )
    
    # Second split: separate train and validation
    train_data, val_data = train_test_split(
        train_val_data, test_size=val_ratio / (1 - test_ratio), random_state=42
    )
    
    # Unpack the data
    train_rgb, train_nir, train_mask = zip(*train_data)
    val_rgb, val_nir, val_mask = zip(*val_data)
    test_rgb, test_nir, test_mask = zip(*test_data)

    # Convert tuples back to lists
    train_rgb, train_nir, train_mask = list(train_rgb), list(train_nir), list(train_mask)
    val_rgb, val_nir, val_mask = list(val_rgb), list(val_nir), list(val_mask)
    test_rgb, test_nir, test_mask = list(test_rgb), list(test_nir), list(test_mask)

    # PIL transforms (geometric augmentations - applied to PIL Images)
    base_transforms = [T.Resize((256, 256))]
    
    if augment:
        base_transforms.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(10),
        ])
    
    pil_transform = T.Compose(base_transforms)

    # Tensor transforms (REMOVED normalization for now to debug)
    tensor_transform = None

    train_dataset = USASegmentationDataset(
        train_rgb, train_nir, train_mask, 
        pil_transform=pil_transform, 
        tensor_transform=tensor_transform
    )
    val_dataset = USASegmentationDataset(
        val_rgb, val_nir, val_mask, 
        pil_transform=T.Resize((256, 256)),
        tensor_transform=None
    )
    test_dataset = USASegmentationDataset(
        test_rgb, test_nir, test_mask, 
        pil_transform=T.Resize((256, 256)),
        tensor_transform=None
    )

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    }


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted CrossEntropy to handle class imbalance"""
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight
        
    def forward(self, outputs, targets):
        return nn.functional.cross_entropy(outputs, targets, weight=self.weight)


def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    correct_pixels = 0
    total_pixels = 0

    for images, masks in tqdm(dataloader, desc="Training"):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)  # (B, C, H, W)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        
        # Calculate accuracy
        pred = torch.argmax(outputs, dim=1)
        correct_pixels += (pred == masks).sum().item()
        total_pixels += masks.numel()

    accuracy = correct_pixels / total_pixels
    return epoch_loss / len(dataloader), accuracy


def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            epoch_loss += loss.item()
            
            # Calculate accuracy
            pred = torch.argmax(outputs, dim=1)
            correct_pixels += (pred == masks).sum().item()
            total_pixels += masks.numel()

    accuracy = correct_pixels / total_pixels
    return epoch_loss / len(dataloader), accuracy


def plot_predictions_fixed(model, dataloader, num_images=5):
    """Fixed plotting function to properly display 4-channel images"""
    model.eval()
    images, masks = next(iter(dataloader))
    images, masks = images.to(device), masks.to(device)

    with torch.no_grad():
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    
    for i in range(num_images):
        # Display only RGB channels (first 3) for visualization
        rgb_img = images[i][:3].cpu().permute(1, 2, 0).numpy()
        rgb_img = np.clip(rgb_img, 0, 1)  # Ensure values are in [0,1]
        
        axes[i, 0].imshow(rgb_img)
        axes[i, 0].set_title('RGB Input')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(masks[i].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(predictions[i].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()


def debug_model_outputs(model, dataloader):
    """Debug function to check model outputs"""
    model.eval()
    images, masks = next(iter(dataloader))
    images, masks = images.to(device), masks.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
        probabilities = torch.softmax(outputs, dim=1)
    
    print(f"Input shape: {images.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Mask shape: {masks.shape}")
    print(f"Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
    print(f"Mask unique values: {torch.unique(masks)}")
    print(f"Prediction unique values: {torch.unique(predictions)}")
    print(f"Dead tree probability range: [{probabilities[:, 1].min():.3f}, {probabilities[:, 1].max():.3f}]")
    
    # Check class distribution in batch
    mask_flat = masks.flatten()
    pred_flat = predictions.flatten()
    print(f"True dead tree pixels: {(mask_flat == 1).sum().item()}/{mask_flat.numel()} ({(mask_flat == 1).float().mean():.3f})")
    print(f"Predicted dead tree pixels: {(pred_flat == 1).sum().item()}/{pred_flat.numel()} ({(pred_flat == 1).float().mean():.3f})")


# Example usage with improvements:
if __name__ == "__main__":
    # Analyze dataset first
    dead_ratio = analyze_dataset("USA_segmentation")
    
    # Create weighted loss based on class distribution
    # If dead trees are 5% of pixels, weight them more heavily
    class_weights = torch.tensor([1.0, 1.0/dead_ratio if dead_ratio > 0 else 10.0])
    print(f"Using class weights: {class_weights}")
    
    # Build data loaders
    loaders = build_loaders("USA_segmentation", batch_size=8, augment=True)
    
    # Setup model and training
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = UNet(in_channels=4, out_channels=2).to(device)
    
    # Use weighted loss
    class_weights = class_weights.to(dtype=torch.float32, device=device)
    criterion = WeightedCrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Increased learning rate
    
    # Debug model before training
    print("\nBefore training:")
    debug_model_outputs(model, loaders['train'])
    
    # Training loop with accuracy tracking
    num_epochs = 10
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, loaders['train'], optimizer, criterion)
        val_loss, val_acc = evaluate(model, loaders['val'], criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")
    
    # Debug model after training
    print("\nAfter training:")
    debug_model_outputs(model, loaders['test'])
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Progress - Loss')
    ax1.legend()
    
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Progress - Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Show predictions
    plot_predictions_fixed(model, loaders['test'], num_images=5)