#!/usr/bin/env python3
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.deeplabv3plus import DeepLabV3Plus
from datasets.multispectral import MultispectralDataset
from utils.metrics import calculate_metrics
from utils.spectral_indices import calculate_indices

logger = logging.getLogger(__name__)

def train_model(config: dict):
    """Train the DeepLabV3+ model on multi-spectral data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Initialize model
    model = DeepLabV3Plus(
        num_classes=config['model']['num_classes'],
        backbone=config['model']['backbone'],
        pretrained=config['model']['pretrained'],
        in_channels=len(config['model']['bands'])
    ).to(device)

    # Setup datasets and dataloaders
    train_dataset = MultispectralDataset(
        data_dir=config['data']['train_dir'],
        input_size=config['data']['input_size'],
        bands=config['model']['bands'],
        indices=config['model']['indices'] if config['model']['use_indices'] else None,
        is_training=True
    )
    
    val_dataset = MultispectralDataset(
        data_dir=config['data']['val_dir'],
        input_size=config['data']['input_size'],
        bands=config['model']['bands'],
        indices=config['model']['indices'] if config['model']['use_indices'] else None,
        is_training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Setup learning rate scheduler
    if config['training']['lr_scheduler']['type'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['lr_scheduler']['params']['T_max'],
            eta_min=config['training']['lr_scheduler']['params']['eta_min']
        )

    # Setup tensorboard
    writer = SummaryWriter(config['training']['log_dir'])
    best_val_iou = 0.0
    checkpoint_dir = Path(config['training']['checkpoint_dir'])

    # Training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}") as pbar:
            for batch_idx, (images, masks) in enumerate(pbar):
                images, masks = images.to(device), masks.to(device)
                
                # Calculate spectral indices if enabled
                if config['model']['use_indices']:
                    indices = calculate_indices(images, config['model']['indices'])
                    images = torch.cat([images, indices], dim=1)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix({'loss': train_loss / (batch_idx + 1)})

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_metrics = {'iou': 0.0, 'pixel_acc': 0.0}
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                
                if config['model']['use_indices']:
                    indices = calculate_indices(images, config['model']['indices'])
                    images = torch.cat([images, indices], dim=1)

                outputs = model(images)
                val_loss += criterion(outputs, masks).item()
                
                # Calculate metrics
                batch_metrics = calculate_metrics(outputs, masks)
                for k, v in batch_metrics.items():
                    val_metrics[k] += v

        val_loss /= len(val_loader)
        for k in val_metrics:
            val_metrics[k] /= len(val_loader)

        # Log metrics
        writer.add_scalar('Loss/train', train_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/IoU', val_metrics['iou'], epoch)
        writer.add_scalar('Metrics/PixelAcc', val_metrics['pixel_acc'], epoch)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': best_val_iou,
            }, checkpoint_dir / 'best_model.pth')
            
        # Save latest model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_iou': val_metrics['iou'],
        }, checkpoint_dir / 'latest_model.pth')

        logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, "
                   f"Val Loss = {val_loss:.4f}, Val IoU = {val_metrics['iou']:.4f}")

    writer.close()
    logger.info("Training completed!")