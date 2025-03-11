"""Metrics calculation for semantic segmentation evaluation.

This module implements common metrics for evaluating semantic segmentation models:
- Pixel Accuracy
- Mean Pixel Accuracy
- Mean Intersection over Union (mIoU)
- Frequency Weighted Intersection over Union (FWIoU)
- Dice Coefficient
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np


class SegmentationMetrics:
    """Calculate and accumulate semantic segmentation metrics."""
    
    def __init__(self, num_classes: int):
        """Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes (including background)
        """
        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros((num_classes, num_classes), 
                                         dtype=torch.int64)
    
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Update confusion matrix with batch predictions.
        
        Args:
            pred: Prediction tensor of shape (B, H, W) with class indices
            target: Ground truth tensor of shape (B, H, W) with class indices
        """
        pred = pred.flatten()
        target = target.flatten()
        
        # Calculate confusion matrix
        mask = (target >= 0) & (target < self.num_classes)
        hist = torch.bincount(
            self.num_classes * target[mask].long() + pred[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        
        self.confusion_matrix += hist
    
    def reset(self) -> None:
        """Reset accumulated statistics."""
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes),
                                         dtype=torch.int64)
    
    def pixel_accuracy(self) -> float:
        """Calculate global pixel accuracy.
        
        Returns:
            Global pixel accuracy as a float between 0 and 1
        """
        correct = torch.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return float(correct / total)
    
    def class_pixel_accuracy(self) -> torch.Tensor:
        """Calculate per-class pixel accuracy.
        
        Returns:
            Tensor of per-class accuracies
        """
        correct = torch.diag(self.confusion_matrix)
        total = self.confusion_matrix.sum(dim=1)
        valid = total > 0
        accuracy = torch.zeros_like(total, dtype=torch.float32)
        accuracy[valid] = correct[valid].float() / total[valid].float()
        return accuracy
    
    def mean_pixel_accuracy(self) -> float:
        """Calculate mean pixel accuracy across classes.
        
        Returns:
            Mean pixel accuracy as a float between 0 and 1
        """
        class_accuracy = self.class_pixel_accuracy()
        return float(class_accuracy.mean())
    
    def intersection_over_union(self) -> torch.Tensor:
        """Calculate per-class IoU scores.
        
        Returns:
            Tensor of per-class IoU scores
        """
        intersection = torch.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(dim=0) + 
                self.confusion_matrix.sum(dim=1) - intersection)
        valid = union > 0
        iou = torch.zeros_like(union, dtype=torch.float32)
        iou[valid] = intersection[valid].float() / union[valid].float()
        return iou
    
    def mean_intersection_over_union(self) -> float:
        """Calculate mean IoU across classes.
        
        Returns:
            Mean IoU as a float between 0 and 1
        """
        iou = self.intersection_over_union()
        return float(iou.mean())
    
    def frequency_weighted_intersection_over_union(self) -> float:
        """Calculate frequency weighted IoU.
        
        Returns:
            Frequency weighted IoU as a float between 0 and 1
        """
        freq = self.confusion_matrix.sum(dim=1) / self.confusion_matrix.sum()
        iou = self.intersection_over_union()
        return float((freq * iou).sum())
    
    def dice_coefficient(self) -> torch.Tensor:
        """Calculate per-class Dice coefficients.
        
        Returns:
            Tensor of per-class Dice coefficients
        """
        intersection = torch.diag(self.confusion_matrix)
        sum_pred_target = (self.confusion_matrix.sum(dim=0) + 
                         self.confusion_matrix.sum(dim=1))
        valid = sum_pred_target > 0
        dice = torch.zeros_like(sum_pred_target, dtype=torch.float32)
        dice[valid] = (2 * intersection[valid].float() / 
                      sum_pred_target[valid].float())
        return dice
    
    def mean_dice_coefficient(self) -> float:
        """Calculate mean Dice coefficient across classes.
        
        Returns:
            Mean Dice coefficient as a float between 0 and 1
        """
        dice = self.dice_coefficient()
        return float(dice.mean())
    
    def get_scores(self) -> Dict[str, float]:
        """Get all metric scores.
        
        Returns:
            Dictionary containing all metric scores
        """
        return {
            'pixel_acc': self.pixel_accuracy(),
            'mean_pixel_acc': self.mean_pixel_accuracy(),
            'mean_iou': self.mean_intersection_over_union(),
            'freq_weighted_iou': self.frequency_weighted_intersection_over_union(),
            'mean_dice': self.mean_dice_coefficient()
        }


def evaluate_batch(pred: torch.Tensor, target: torch.Tensor, 
                  num_classes: int) -> Dict[str, float]:
    """Evaluate a single batch of predictions.
    
    Args:
        pred: Prediction tensor of shape (B, H, W) with class indices
        target: Ground truth tensor of shape (B, H, W) with class indices
        num_classes: Number of classes (including background)
    
    Returns:
        Dictionary containing metric scores for the batch
    """
    metrics = SegmentationMetrics(num_classes)
    metrics.update(pred, target)
    return metrics.get_scores()


def evaluate_predictions(pred: torch.Tensor, target: torch.Tensor, 
                       num_classes: int, 
                       ignore_index: Optional[int] = None) -> Dict[str, float]:
    """Evaluate predictions with optional ignore index.
    
    Args:
        pred: Prediction tensor of shape (B, C, H, W) or (B, H, W)
        target: Ground truth tensor of shape (B, H, W)
        num_classes: Number of classes (including background)
        ignore_index: Optional index to ignore in evaluation
    
    Returns:
        Dictionary containing metric scores
    """
    if pred.dim() == 4:
        # Convert probabilities to class indices
        pred = pred.argmax(dim=1)
    
    if ignore_index is not None:
        mask = target != ignore_index
        pred = pred[mask]
        target = target[mask]
    
    return evaluate_batch(pred, target, num_classes)