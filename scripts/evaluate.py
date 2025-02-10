import os
import torch
import argparse
from model import SwinTransformerSegModel
from losses import ComboLoss, dice_coefficient, iou_coefficient, compute_additional_metrics, compute_confusion_matrix
from utils import create_dataloaders
from data import ChestXRayDataset, JointTransformWrapper

def evaluate_model(model, test_loader, device):
    model.eval()
    criterion = ComboLoss()  # Use the same combo loss
    test_loss = 0.0
    test_dice = 0.0
    test_iou = 0.0
    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item() * images.size(0)
            d = dice_coefficient(outputs, masks)
            i = iou_coefficient(outputs, masks)
            test_dice += d * images.size(0)
            test_iou += i * images.size(0)
            tp, fp, tn, fn = compute_confusion_matrix(outputs, masks)
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
    
    test_loss /= len(test_loader.dataset)
    test_dice /= len(test_loader.dataset)
    test_iou /= len(test_loader.dataset)
    metrics = compute_additional_metrics(total_tp, total_fp, total_tn, total_fn)
    
    print("=== Test Results ===")
    print(f"Loss: {test_loss:.4f}")
    print(f"Dice: {test_dice:.4f} | IoU: {test_iou:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1: {metrics['f1_score']:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Chest X-Ray Segmentation Model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model weights")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    args = parser.parse_args()
    
    images_dir = os.path.join(args.dataset_path, "images")
    masks_dir = os.path.join(args.dataset_path, "masks")
    
    joint_transform = JointTransformWrapper(augment=False, image_size=(224, 224))
    full_dataset = ChestXRayDataset(images_dir, masks_dir, transform=joint_transform)
    
    dataset_len = len(full_dataset)
    train_size = int(0.7 * dataset_len)
    val_size = int(0.15 * dataset_len)
    test_size = dataset_len - train_size - val_size
    _, _, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])
    
    # Create dataloader for test set.
    _, _, test_loader = create_dataloaders([], [], test_dataset, batch_size=args.batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinTransformerSegModel(backbone_name="swin_tiny_patch4_window7_224", out_channels=1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()
