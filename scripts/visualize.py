import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from data import ChestXRayDataset, JointTransformWrapper
from model import SwinTransformerSegModel

def overlay_mask_on_image(image, mask, alpha=0.5, color=(0, 255, 0)):
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def visualize_predictions(model, loader, device, num_samples=4):
    model.eval()
    batch = next(iter(loader))
    images, masks = batch
    images = images.to(device)
    masks = masks.to(device)
    with torch.no_grad():
        outputs = model(images)
        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()
    for i in range(min(num_samples, images.size(0))):
        img_np = images[i].cpu().numpy().transpose(1, 2, 0)
        mask_gt = masks[i].cpu().numpy().squeeze()
        mask_pred = preds[i].cpu().numpy().squeeze()
        
        # Denormalize image using ImageNet stats.
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = (img_np * std + mean)
        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)
        
        overlay_gt = overlay_mask_on_image(img_np, mask_gt, alpha=0.5, color=(255, 0, 0))
        overlay_pred = overlay_mask_on_image(img_np, mask_pred, alpha=0.5, color=(0, 255, 0))
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 5))
        axes[0].imshow(img_np)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(overlay_gt)
        axes[1].set_title("Ground Truth Overlay (Red)")
        axes[1].axis("off")
        axes[2].imshow(overlay_pred)
        axes[2].set_title("Prediction Overlay (Green)")
        axes[2].axis("off")
        plt.tight_layout()
        plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize Predictions")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    args = parser.parse_args()
    
    images_dir = os.path.join(args.dataset_path, "images")
    masks_dir = os.path.join(args.dataset_path, "masks")
    
    joint_transform = JointTransformWrapper(augment=False, image_size=(224, 224))
    dataset = ChestXRayDataset(images_dir, masks_dir, transform=joint_transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinTransformerSegModel(backbone_name="swin_tiny_patch4_window7_224", out_channels=1).to(device)
    # Optionally, load trained weights here:
    # model.load_state_dict(torch.load("path_to_weights.pth", map_location=device))
    model.eval()
    
    visualize_predictions(model, loader, device, num_samples=4)

if __name__ == "__main__":
    main()
