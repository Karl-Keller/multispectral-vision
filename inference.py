#!/usr/bin/env python3
import logging
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.deeplabv3plus import DeepLabV3Plus
from datasets.multispectral import MultispectralDataset
from utils.spectral_indices import calculate_indices
from utils.visualization import create_segmentation_overlay

logger = logging.getLogger(__name__)

def run_inference(config: dict):
    """Run inference using a trained DeepLabV3+ model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Initialize model
    model = DeepLabV3Plus(
        num_classes=config['model']['num_classes'],
        backbone=config['model']['backbone'],
        pretrained=False,  # Don't need pretrained for inference
        in_channels=len(config['model']['bands'])
    ).to(device)

    # Load checkpoint
    checkpoint_path = Path(config['inference']['checkpoint'])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Setup inference dataset and dataloader
    inference_dataset = MultispectralDataset(
        data_dir=config['inference']['input_dir'],
        input_size=config['data']['input_size'],
        bands=config['model']['bands'],
        indices=config['model']['indices'] if config['model']['use_indices'] else None,
        is_training=False
    )

    inference_loader = DataLoader(
        inference_dataset,
        batch_size=config['inference']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    # Create output directory
    output_dir = Path(config['inference']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run inference
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(inference_loader, desc="Running inference")):
            images = images.to(device)
            
            # Calculate spectral indices if enabled
            if config['model']['use_indices']:
                indices = calculate_indices(images, config['model']['indices'])
                images = torch.cat([images, indices], dim=1)

            # Get predictions
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            # Save predictions
            for i, pred in enumerate(predictions):
                pred_np = pred.cpu().numpy()
                
                # Save raw predictions
                pred_path = output_dir / f"prediction_{batch_idx * config['inference']['batch_size'] + i}.npy"
                np.save(pred_path, pred_np)

                # Create and save visualization if enabled
                if config['inference']['save_visualization']:
                    # Get RGB channels for visualization (assuming they're in the right order)
                    rgb_idx = [config['model']['bands'].index(band) for band in ['red', 'green', 'blue']]
                    rgb_image = images[i, rgb_idx].cpu().numpy()
                    
                    # Create visualization
                    vis_path = output_dir / f"vis_{batch_idx * config['inference']['batch_size'] + i}.png"
                    create_segmentation_overlay(rgb_image, pred_np, vis_path)

    logger.info(f"Inference completed! Results saved to {output_dir}")