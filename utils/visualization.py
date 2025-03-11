import os
from pathlib import Path
import logging
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class VisualizationError(Exception):
    """Custom exception for visualization errors."""
    pass

def create_segmentation_overlay(
    image: np.ndarray,
    prediction: np.ndarray,
    save_path: Union[str, Path],
    create_dir: bool = True
) -> Optional[str]:
    """Create and save a visualization of the segmentation results.
    
    Args:
        image: RGB image array of shape [3, H, W]
        prediction: Segmentation mask of shape [H, W]
        save_path: Path to save the visualization
        create_dir: Whether to create the directory if it doesn't exist
        
    Returns:
        Path to saved visualization if successful, None otherwise
        
    Raises:
        VisualizationError: If there's an error during visualization
    """
    try:
        # Input validation
        if not isinstance(image, np.ndarray) or not isinstance(prediction, np.ndarray):
            raise VisualizationError("Image and prediction must be numpy arrays")
            
        if image.shape[0] != 3 or len(image.shape) != 3:
            raise VisualizationError(f"Invalid image shape: {image.shape}, expected (3, H, W)")
            
        if len(prediction.shape) != 2:
            raise VisualizationError(f"Invalid prediction shape: {prediction.shape}, expected (H, W)")
            
        # Ensure save directory exists
        save_path = Path(save_path)
        if create_dir:
            save_path.parent.mkdir(parents=True, exist_ok=True)

        # Normalize RGB image
        rgb = np.moveaxis(image, 0, -1)  # Change to [H, W, 3]
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        
        # Create color map for segmentation
        num_classes = prediction.max() + 1
        colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
        
        # Create segmentation overlay
        seg_image = colors[prediction]
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original image
        ax1.imshow(rgb)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Plot segmentation
        ax2.imshow(seg_image)
        ax2.set_title('Segmentation')
        ax2.axis('off')
        
        # Plot overlay
        overlay = rgb * 0.7 + seg_image[:, :, :3] * 0.3
        ax3.imshow(overlay)
        ax3.set_title('Overlay')
        ax3.axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        raise VisualizationError(f"Failed to create visualization: {str(e)}")