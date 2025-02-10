import os
import copy
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from data import ChestXRayDataset, JointTransformWrapper
from model import SwinTransformerSegModel
from losses import ComboLoss, dice_coefficient, iou_coefficient, compute_additional_metrics, compute_confusion_matrix
from utils import format_time, create_dataloaders

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(train_loader.dataset)

def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)
            d = dice_coefficient(outputs, masks)
            i = iou_coefficient(outputs, masks)
            running_dice += d * images.size(0)
            running_iou += i * images.size(0)
            tp, fp, tn, fn = compute_confusion_matrix(outputs, masks)
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
    metrics = compute_additional_metrics(total_tp, total_fp, total_tn, total_fn)
    return (running_loss / len(val_loader.dataset),
            running_dice / len(val_loader.dataset),
            running_iou / len(val_loader.dataset),
            metrics)

def plot_training_curves(history, output_dir):
    epochs_range = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(16, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, history["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(2, 3, 2)
    plt.plot(epochs_range, history["val_dice"], label="Val Dice")
    plt.title("Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.legend()
    
    plt.subplot(2, 3, 3)
    plt.plot(epochs_range, history["val_iou"], label="Val IoU")
    plt.title("IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()
    
    plt.subplot(2, 3, 4)
    plt.plot(epochs_range, history["val_accuracy"], label="Val Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.subplot(2, 3, 5)
    plt.plot(epochs_range, history["val_precision"], label="Val Precision")
    plt.plot(epochs_range, history["val_recall"], label="Val Recall")
    plt.title("Precision & Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    
    plt.subplot(2, 3, 6)
    plt.plot(epochs_range, history["val_specificity"], label="Val Specificity")
    plt.plot(epochs_range, history["val_f1"], label="Val F1")
    plt.title("Specificity & F1")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    plt.close()

def train_model(model, train_loader, val_loader, device, epochs=20, patience=5, lr=1e-4, weight_bce=0.5, reduce_on_plateau=True):
    criterion = ComboLoss(weight_bce=weight_bce)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = None
    if reduce_on_plateau:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": [],
               "val_accuracy": [], "val_precision": [], "val_recall": [],
               "val_specificity": [], "val_f1": []}
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    no_improve_count = 0
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_iou, metrics = validate_one_epoch(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        history["val_accuracy"].append(metrics["accuracy"])
        history["val_precision"].append(metrics["precision"])
        history["val_recall"].append(metrics["recall"])
        history["val_specificity"].append(metrics["specificity"])
        history["val_f1"].append(metrics["f1_score"])
        
        if scheduler:
            scheduler.step(val_loss)
            current_lr = scheduler.optimizer.param_groups[0]['lr']
        else:
            current_lr = lr
        
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        total_time = epoch_end - start_time
        avg_epoch_time = total_time / epoch
        remaining_time = avg_epoch_time * (epochs - epoch)
        
        print(f"Epoch [{epoch}/{epochs}] (LR: {current_lr:.6f})")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, Specificity: {metrics['specificity']:.4f}, F1: {metrics['f1_score']:.4f}")
        print(f"  Time Elapsed: {format_time(total_time)} | Epoch Time: {format_time(epoch_time)} | Est. Remaining: {format_time(remaining_time)}\n")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= patience:
            print("Early stopping triggered!")
            break
    
    model.load_state_dict(best_model_wts)
    return model, history

def main():
    parser = argparse.ArgumentParser(description="Train Chest X-Ray Segmentation Model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs and logs")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define dataset paths.
    images_dir = os.path.join(args.dataset_path, "images")
    masks_dir = os.path.join(args.dataset_path, "masks")
    
    # Create transformation pipeline.
    joint_transform = JointTransformWrapper(augment=True, image_size=(224, 224))
    
    # Create dataset.
    full_dataset = ChestXRayDataset(images_dir, masks_dir, transform=joint_transform)
    print("Total samples in dataset:", len(full_dataset))
    
    # Split dataset.
    dataset_len = len(full_dataset)
    train_size = int(0.7 * dataset_len)
    val_size = int(0.15 * dataset_len)
    test_size = dataset_len - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])
    
    # Create dataloaders.
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=args.batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Initialize model.
    model = SwinTransformerSegModel(backbone_name="swin_tiny_patch4_window7_224", out_channels=1).to(device)
    
    # Train the model.
    model, history = train_model(model, train_loader, val_loader, device,
                                 epochs=args.epochs, patience=args.patience, lr=args.lr)
    
    # Save model weights.
    model_save_path = os.path.join(args.output_dir, "swin_segmentation_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Plot training curves.
    plot_training_curves(history, args.output_dir)

if __name__ == "__main__":
    main()
