import io
from typing import List, Tuple, Dict, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision.models._utils import _ModelTransform

def get_model_summary(model: torch.nn.Module) -> str:
    """Generate a string summary of model architecture.
    
    Args:
        model: PyTorch model
        
    Returns:
        String containing model summary
    """
    def get_layer_info(layer: torch.nn.Module) -> Tuple[int, int]:
        """Get number of parameters and operations for a layer."""
        params = sum(p.numel() for p in layer.parameters())
        # Basic estimate of multiply-adds for the layer
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            ops = (params * layer.output_size[2] * layer.output_size[3])
        elif isinstance(layer, torch.nn.Linear):
            ops = params
        else:
            ops = 0
        return params, ops

    # Capture model forward pass to get layer shapes
    def hook_fn(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
        if not hasattr(module, 'output_size'):
            module.output_size = output.shape

    hooks = []
    for layer in model.modules():
        hooks.append(layer.register_forward_hook(hook_fn))

    # Run a forward pass with dummy data
    try:
        dummy_input = torch.randn(1, model.backbone.conv1.in_channels, 224, 224)
        model(dummy_input)
    except Exception as e:
        print(f"Warning: Forward pass failed: {e}")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Generate summary
    summary = []
    summary.append("Model Architecture Summary")
    summary.append("=" * 80)
    summary.append(f"{'Layer':<40} {'Output Shape':<20} {'Params':<10} {'Memory (MB)':<15}")
    summary.append("-" * 80)

    total_params = 0
    total_memory = 0

    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
            params = sum(p.numel() for p in layer.parameters())
            memory = params * 4 / (1024 * 1024)  # 4 bytes per parameter
            output_shape = getattr(layer, 'output_size', ['?'])
            shape_str = str(list(output_shape)) if output_shape else '?'
            
            summary.append(f"{name:<40} {shape_str:<20} {params:<10,d} {memory:<15.2f}")
            
            total_params += params
            total_memory += memory

    summary.append("-" * 80)
    summary.append(f"Total Parameters: {total_params:,}")
    summary.append(f"Total Memory: {total_memory:.2f} MB")
    summary.append("=" * 80)

    return "\n".join(summary)

def compute_confusion_matrix(
    model: torch.nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    device: torch.device,
    use_indices: bool = False,
    indices_list: Optional[List[str]] = None
) -> np.ndarray:
    """Compute confusion matrix for the model.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for validation/test data
        num_classes: Number of classes
        device: Device to run inference on
        use_indices: Whether to use spectral indices
        indices_list: List of spectral indices to compute
        
    Returns:
        Confusion matrix as numpy array
    """
    confusion_matrix = np.zeros((num_classes, num_classes))
    model.eval()

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            if use_indices:
                from utils.spectral_indices import calculate_indices
                indices = calculate_indices(images, indices_list)
                images = torch.cat([images, indices], dim=1)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            # Update confusion matrix
            for t, p in zip(targets.view(-1), predictions.view(-1)):
                confusion_matrix[t.item(), p.item()] += 1

    return confusion_matrix

def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """Create a confusion matrix visualization.
    
    Args:
        confusion_matrix: Numpy array of confusion matrix
        class_names: List of class names
        normalize: Whether to normalize the confusion matrix
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if normalize:
        confusion_matrix = (
            confusion_matrix.astype('float') / 
            confusion_matrix.sum(axis=1)[:, np.newaxis]
        )
        
    if class_names is None:
        class_names = [str(i) for i in range(confusion_matrix.shape[0])]
        
    plt.figure(figsize=figsize)
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return buf