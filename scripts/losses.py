import torch
import torch.nn as nn

def dice_loss(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = 1 - (2. * intersection + smooth) / (union + smooth)
    return dice.mean()

class ComboLoss(nn.Module):
    """
    Weighted combination of BCEWithLogitsLoss and Dice Loss.
    """
    def __init__(self, weight_bce=0.5):
        super(ComboLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.weight_bce = weight_bce
        
    def forward(self, pred, target):
        loss_bce = self.bce(pred, target)
        loss_dice = dice_loss(pred, target)
        return self.weight_bce * loss_bce + (1 - self.weight_bce) * loss_dice

def dice_coefficient(pred, target, threshold=0.5, smooth=1e-5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def iou_coefficient(pred, target, threshold=0.5, smooth=1e-5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

def compute_confusion_matrix(pred, target, threshold=0.5):
    """
    Computes pixel-level confusion matrix (TP, FP, TN, FN) for a batch.
    """
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum()
    return tp.item(), fp.item(), tn.item(), fn.item()

def compute_additional_metrics(tp, fp, tn, fn, eps=1e-7):
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1
    }
