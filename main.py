#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import yaml

from train import train_model
from inference import run_inference

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_directories(config: dict):
    """Create necessary directories if they don't exist."""
    dirs = []
    
    # Common directories
    output_dir = Path(config.get('output_dir', 'outputs'))
    dirs.append(output_dir)
    
    # Training mode directories
    if config.get('mode') == 'train':
        data_dir = Path(config['data']['data_dir'])
        dirs.extend([
            data_dir / 'train',
            data_dir / 'val',
            data_dir / 'test',
            output_dir / 'checkpoints',
            output_dir / 'logs'
        ])
    
    # Processing mode directories
    elif config.get('mode') == 'process':
        data_dir = Path(config['data']['data_dir'])
        dirs.extend([
            data_dir,
            output_dir / 'predictions',
            output_dir / 'visualizations'
        ])
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def main():
    parser = argparse.ArgumentParser(description='Multi-spectral Vision Processor')
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    parser.add_argument('--mode', type=str, choices=['train', 'process'], required=True,
                      help='Operation mode: train or process')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Setup directories
    setup_directories(config)
    
    # Run requested mode
    if args.mode == 'train':
        logger.info("Starting training mode...")
        train_model(config)
    else:
        logger.info("Starting inference mode...")
        run_inference(config)

if __name__ == '__main__':
    main()