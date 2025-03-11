#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path

from train import train_model
from inference import run_inference

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def setup_directories(config: dict):
    """Create necessary directories if they don't exist."""
    dirs = [
        Path(config['data']['train_dir']),
        Path(config['data']['val_dir']),
        Path(config['data']['test_dir']),
        Path(config['training']['checkpoint_dir']),
        Path(config['training']['log_dir']),
        Path(config['inference']['input_dir']),
        Path(config['inference']['output_dir'])
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def main():
    parser = argparse.ArgumentParser(description='Multi-spectral Vision Processor')
    parser.add_argument('--config', type=str, required=True, help='Path to config.json')
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], required=True,
                      help='Operation mode: train or inference')
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