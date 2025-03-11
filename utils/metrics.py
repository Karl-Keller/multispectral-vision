import torch

def calculate_metrics(outputs: torch.Tensor, targets: torch.Tensor) -> dict:
    """Calculate segmentation metrics.
    
    Args:
        outputs: Model predictions [B, C, H, W]
        targets: Ground truth labels [B, H, W]
        
    Returns:
        Dictionary containing metrics
    """
    predictions = torch.argmax(outputs, dim=1)
    
    # Calculate IoU
    num_classes = outputs.size(1)
    iou_sum = 0.0
    
    for cls in range(num_classes):
        pred_mask = (predictions == cls)
        target_mask = (targets == cls)
        
        intersection = (pred_mask & target_mask).float().sum()
        union = (pred_mask | target_mask).float().sum()
        
        if union > 0:
            iou_sum += intersection / union
    
    mean_iou = iou_sum / num_classes
    
    # Calculate pixel accuracy
    correct = (predictions == targets).float().sum()
    total = torch.numel(targets)
    pixel_acc = correct / total
    
    return {
        'iou': mean_iou.item(),
        'pixel_acc': pixel_acc.item()
    }