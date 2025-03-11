"""Tests for segmentation metrics calculation."""

import pytest
import torch
import numpy as np

from src.metrics import (
    SegmentationMetrics,
    evaluate_batch,
    evaluate_predictions
)


@pytest.fixture
def binary_case():
    """Create a simple binary segmentation case."""
    pred = torch.tensor([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 1, 1]
    ])
    target = torch.tensor([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1]
    ])
    return pred, target


@pytest.fixture
def multiclass_case():
    """Create a multiclass segmentation case."""
    pred = torch.tensor([
        [0, 0, 1, 2],
        [0, 1, 1, 2],
        [0, 1, 2, 2],
        [0, 0, 1, 2]
    ])
    target = torch.tensor([
        [0, 0, 1, 2],
        [0, 0, 1, 2],
        [0, 0, 2, 2],
        [0, 0, 1, 2]
    ])
    return pred, target


def test_confusion_matrix(binary_case):
    """Test confusion matrix calculation."""
    pred, target = binary_case
    metrics = SegmentationMetrics(num_classes=2)
    metrics.update(pred, target)
    
    expected_matrix = torch.tensor([
        [6, 1],  # 6 correct 0s, 1 incorrect (predicted 1, true 0)
        [0, 9]   # 9 correct 1s, 0 incorrect
    ])
    
    assert torch.all(metrics.confusion_matrix == expected_matrix)


def test_pixel_accuracy(binary_case):
    """Test pixel accuracy calculation."""
    pred, target = binary_case
    metrics = SegmentationMetrics(num_classes=2)
    metrics.update(pred, target)
    
    # 15 correct predictions out of 16 total pixels
    expected_accuracy = 15 / 16
    assert abs(metrics.pixel_accuracy() - expected_accuracy) < 1e-6


def test_class_pixel_accuracy(multiclass_case):
    """Test per-class pixel accuracy."""
    pred, target = multiclass_case
    metrics = SegmentationMetrics(num_classes=3)
    metrics.update(pred, target)
    
    class_acc = metrics.class_pixel_accuracy()
    
    # Class 0: 6 correct out of 8
    # Class 1: 3 correct out of 4
    # Class 2: 4 correct out of 4
    expected = torch.tensor([6/8, 3/4, 4/4])
    
    assert torch.allclose(class_acc, expected, atol=1e-6)


def test_mean_pixel_accuracy(multiclass_case):
    """Test mean pixel accuracy calculation."""
    pred, target = multiclass_case
    metrics = SegmentationMetrics(num_classes=3)
    metrics.update(pred, target)
    
    # Mean of per-class accuracies: (6/8 + 3/4 + 4/4) / 3
    expected_mean = (6/8 + 3/4 + 4/4) / 3
    assert abs(metrics.mean_pixel_accuracy() - expected_mean) < 1e-6


def test_intersection_over_union(binary_case):
    """Test IoU calculation."""
    pred, target = binary_case
    metrics = SegmentationMetrics(num_classes=2)
    metrics.update(pred, target)
    
    iou = metrics.intersection_over_union()
    
    # Class 0: intersection=6, union=7
    # Class 1: intersection=9, union=10
    expected = torch.tensor([6/7, 9/10])
    
    assert torch.allclose(iou, expected, atol=1e-6)


def test_mean_intersection_over_union(multiclass_case):
    """Test mean IoU calculation."""
    pred, target = multiclass_case
    metrics = SegmentationMetrics(num_classes=3)
    metrics.update(pred, target)
    
    miou = metrics.mean_intersection_over_union()
    
    # Mean of per-class IoUs
    iou = metrics.intersection_over_union()
    expected_mean = float(iou.mean())
    
    assert abs(miou - expected_mean) < 1e-6


def test_frequency_weighted_iou(multiclass_case):
    """Test frequency weighted IoU calculation."""
    pred, target = multiclass_case
    metrics = SegmentationMetrics(num_classes=3)
    metrics.update(pred, target)
    
    fwiou = metrics.frequency_weighted_intersection_over_union()
    
    # Should be weighted by class frequencies
    assert 0 <= fwiou <= 1


def test_dice_coefficient(binary_case):
    """Test Dice coefficient calculation."""
    pred, target = binary_case
    metrics = SegmentationMetrics(num_classes=2)
    metrics.update(pred, target)
    
    dice = metrics.dice_coefficient()
    
    # Class 0: 2*6 / (7+8)
    # Class 1: 2*9 / (10+10)
    expected = torch.tensor([2*6/(7+8), 2*9/(10+10)])
    
    assert torch.allclose(dice, expected, atol=1e-6)


def test_evaluate_batch(multiclass_case):
    """Test batch evaluation function."""
    pred, target = multiclass_case
    scores = evaluate_batch(pred, target, num_classes=3)
    
    assert isinstance(scores, dict)
    assert all(0 <= v <= 1 for v in scores.values())
    assert set(scores.keys()) == {
        'pixel_acc', 'mean_pixel_acc', 'mean_iou',
        'freq_weighted_iou', 'mean_dice'
    }


def test_evaluate_predictions_with_probabilities():
    """Test evaluation with probability predictions."""
    # Create probability predictions (B, C, H, W)
    probs = torch.zeros((1, 3, 2, 2))
    probs[0, 0, 0, 0] = 1  # Class 0
    probs[0, 1, 0, 1] = 1  # Class 1
    probs[0, 2, 1, 0] = 1  # Class 2
    probs[0, 0, 1, 1] = 1  # Class 0
    
    target = torch.tensor([[0, 1], [2, 0]])
    
    scores = evaluate_predictions(probs, target, num_classes=3)
    assert isinstance(scores, dict)
    assert all(0 <= v <= 1 for v in scores.values())


def test_evaluate_predictions_with_ignore_index():
    """Test evaluation with ignore index."""
    pred = torch.tensor([[0, 1], [2, 0]])
    target = torch.tensor([[0, -1], [2, 0]])  # -1 as ignore index
    
    scores = evaluate_predictions(pred, target, num_classes=3, ignore_index=-1)
    assert isinstance(scores, dict)
    assert all(0 <= v <= 1 for v in scores.values())