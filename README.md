# Multi-Spectral DeepLabV3+

A PyTorch implementation of DeepLabV3+ architecture modified for multi-spectral image processing. This implementation extends the powerful semantic segmentation capabilities of DeepLabV3+ to handle multi-spectral imagery by incorporating band attention mechanisms and flexible input channels, with support for both single and dual-source (RGB + MS) imagery.

## Features

- **Dual-Source Architecture**:
  - High-quality RGB stream processing
  - Multi-spectral stream with 11 bands support
  - Adaptive fusion module for quality-aware feature combination
  - Fallback modes for single-source inputs

- **Multi-spectral Input Support**: 
  - Handles 11 standard spectral bands (Coastal, RGB, Red Edge 1-3, NIR, SWIR1-2, Thermal)
  - Flexible architecture adaptable to available bands
  
- **MLOps Integration**:
  - MLflow experiment tracking with SQLite backend
  - Automated metrics logging and model versioning
  - Experiment organization with custom tags
  - Artifact management and model registry
- **Spectral Indices**:
  - NDVI (Normalized Difference Vegetation Index)
  - EVI (Enhanced Vegetation Index)
  - NDWI (Normalized Difference Water Index)
  - NDBI (Normalized Difference Built-up Index)
  - SAVI (Soil Adjusted Vegetation Index)
  - NDRE (Normalized Difference Red Edge)
  - Moisture Index
- **Band Attention**: Adaptive weighting of spectral bands using attention mechanisms
- **DeepLabV3+ Architecture**: 
  - Modified ResNet backbone
  - Atrous Spatial Pyramid Pooling (ASPP)
  - Encoder-decoder structure with skip connections
- **Configurable Parameters**: 
  - Enable/disable spectral indices
  - Adjustable number of output classes
  - Customizable band weights

## Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

### MLflow Setup

The project uses MLflow with SQLite backend for experiment tracking. The default configuration:

```python
tracking_uri: "sqlite:///mlflow.db"
artifact_location: "artifacts"
```

To view the MLflow UI:

```bash
# Start MLflow server
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Access UI at http://localhost:5000
```

### Environment Variables

```bash
# Optional: Set custom MLflow tracking URI
export MLFLOW_TRACKING_URI=sqlite:///path/to/mlflow.db

# Optional: Set custom artifact location
export MLFLOW_ARTIFACT_LOCATION=/path/to/artifacts
```

## Usage

### Basic Usage

The system operates in two distinct modes:
1. **Training Mode**: For model training and evaluation
2. **Processing Mode**: For inference on new data

#### Training Mode

```python
import torch
from models import DualSourceDeepLabV3Plus
from config import Config, TrainingConfig
from spectral_bands import SpectralIndices

# Load training configuration
config = Config.from_yaml("configs/train.yaml")
assert config.mode == "train"

# Initialize model
model = DualSourceDeepLabV3Plus(
    num_classes=config.model.num_classes,
    rgb_encoder=config.model.backbone,
    ms_encoder=config.model.backbone,
    ms_in_channels=len(config.data.bands),
    fusion_channels=256
)

# Training loop
trainer = Trainer(model, config)
trainer.train()
```

Example training configuration (train.yaml):
```yaml
mode: train
experiment_name: multispectral_segmentation
seed: 42

model:
  backbone: resnet50
  output_stride: 16
  num_classes: 21
  pretrained: true

data:
  data_dir: data/dataset
  train_split: train
  val_split: val
  batch_size: 16
  bands: [red, green, blue, nir]

spectral:
  indices: [ndvi, evi]
  use_as_features: true

training:
  epochs: 100
  learning_rate: 0.001
  lr_scheduler: cosine
  mixed_precision: true
```

#### Processing Mode

```python
import torch
from models import DualSourceDeepLabV3Plus
from config import Config
from utils import process_image, save_prediction

# Load processing configuration
config = Config.from_yaml("configs/process.yaml")
assert config.mode == "process"

# Load pretrained model
model = DualSourceDeepLabV3Plus.load_from_checkpoint(
    config.model.checkpoint_path
)
model.eval()

# Process single image
rgb_img = "data/test/rgb_image.tif"
ms_img = "data/test/ms_image.tif"

with torch.no_grad():
    # Process with both RGB and MS data
    prediction = process_image(
        model, rgb_img, ms_img,
        config.processing.batch_size,
        config.processing.half_precision
    )
    
    # Save prediction
    save_prediction(
        prediction,
        "output/prediction.tif",
        config.processing.output_format,
        config.processing.preserve_metadata
    )
```

Example processing configuration (process.yaml):
```yaml
mode: process
experiment_name: inference_run
seed: 42

model:
  backbone: resnet50
  num_classes: 21
  checkpoint_path: models/checkpoint.pth

data:
  data_dir: data/test
  bands: [red, green, blue, nir]

spectral:
  indices: [ndvi, evi]
  use_as_features: true

processing:
  batch_size: 1
  device: cuda
  half_precision: true
  output_format: geotiff
  preserve_metadata: true
  confidence_threshold: 0.5
```
```

### Model Architecture

The model consists of several key components:

1. **Dual-Source Processing**
   - RGB Stream:
     * High-quality RGB image processing
     * Standard ResNet backbone
     * Optimized for visual features
   - MS Stream:
     * Multi-spectral band processing
     * Modified first layer for MS input
     * Band attention mechanisms

2. **Adaptive Fusion Module**
   - Quality-aware feature weighting
   - Cross-modal attention mechanism
   - Adaptive to missing inputs
   - Learnable fusion weights

3. **Band Attention Module**
   - Learns to weight different spectral bands adaptively
   - Uses squeeze-and-excitation mechanism
   - Channel-wise attention for MS bands

4. **Modified ResNet Backbone**
   - Adapted first layer for multi-spectral input
   - Maintains powerful feature extraction capabilities
   - Separate encoders for RGB and MS

5. **ASPP Module**
   - Multiple atrous convolutions at different rates
   - Global average pooling branch
   - Enables multi-scale feature extraction

6. **Decoder**
   - Fuses high-level and low-level features
   - Produces detailed segmentation maps
   - Skip connections from both streams

## Model Parameters

### DualSourceDeepLabV3Plus Parameters

- `num_classes`: Number of output segmentation classes (default: 21)
- `rgb_encoder`: Backbone for RGB stream (default: "resnet50")
- `ms_encoder`: Backbone for MS stream (default: "resnet50")
- `ms_in_channels`: Number of multi-spectral bands (default: 8)
- `fusion_channels`: Number of channels in fusion module (default: 256)
- `pretrained`: Whether to use pretrained encoders (default: True)

### MLflow Configuration

- `experiment_name`: Name of the MLflow experiment (default: "multispectral_segmentation")
- `tracking_uri`: URI for MLflow tracking server (default: "sqlite:///mlflow.db")
- `artifact_location`: Location for storing artifacts (default: "artifacts")
- `tags`: Custom tags for experiment organization

## Example Output Shapes

```python
# Dual-source mode
RGB input:   torch.Size([2, 3, 512, 512])    # High-quality RGB
MS input:    torch.Size([2, 8, 512, 512])    # Multi-spectral
Output:      torch.Size([2, 21, 512, 512])   # Segmentation masks

# Single-source mode (MS only)
MS input:    torch.Size([2, 8, 512, 512])    # Multi-spectral
Output:      torch.Size([2, 21, 512, 512])   # Segmentation masks

# MLflow artifacts
Model state: artifacts/model/model.pth
Metrics:     artifacts/metrics/metrics.json
Plots:       artifacts/plots/*.png
```

## Spectral Bands

The model supports the following spectral bands:

| Band Name | Index | Wavelength (nm) | Common Applications |
|-----------|-------|----------------|-------------------|
| Coastal | 0 | ~443 | Atmospheric correction, water quality |
| Blue | 1 | ~490 | Bathymetry, chlorophyll |
| Green | 2 | ~560 | Peak vegetation reflectance |
| Red | 3 | ~665 | Vegetation absorption |
| Red Edge 1 | 4 | ~705 | Vegetation stress |
| Red Edge 2 | 5 | ~740 | Chlorophyll content |
| Red Edge 3 | 6 | ~783 | Leaf structure |
| NIR | 7 | ~842 | Biomass, LAI |
| SWIR1 | 8 | ~1610 | Moisture content |
| SWIR2 | 9 | ~2190 | Mineral mapping |
| Thermal | 10 | ~10900 | Temperature, heat mapping |

## Spectral Indices

The model calculates the following spectral indices:

| Index | Description | Formula | Application |
|-------|-------------|---------|-------------|
| NDVI | Normalized Difference Vegetation Index | (NIR - RED) / (NIR + RED) | Vegetation health |
| EVI | Enhanced Vegetation Index | G * (NIR - RED) / (NIR + C1*RED - C2*BLUE + L) | Improved vegetation monitoring |
| NDWI | Normalized Difference Water Index | (GREEN - NIR) / (GREEN + NIR) | Water body mapping |
| NDBI | Normalized Difference Built-up Index | (SWIR1 - NIR) / (SWIR1 + NIR) | Urban area mapping |
| SAVI | Soil Adjusted Vegetation Index | ((NIR - RED) / (NIR + RED + L)) * (1 + L) | Vegetation in sparse areas |
| NDRE | Normalized Difference Red Edge | (NIR - RED_EDGE) / (NIR + RED_EDGE) | Crop health monitoring |
| Moisture | Normalized Difference Moisture Index | (NIR - SWIR1) / (NIR + SWIR1) | Vegetation water content |

## File Structure

```
.
├── README.md
├── requirements.txt
├── pyproject.toml                  # Poetry project configuration
├── models/
│   ├── __init__.py
│   ├── dual_source_deeplabv3.py   # Dual-source DeepLabV3+ implementation
│   ├── attention.py               # Band attention modules
│   └── fusion.py                  # Feature fusion modules
├── config/
│   ├── __init__.py
│   ├── base.py                    # Base configuration classes
│   ├── train.py                   # Training configuration
│   └── process.py                 # Processing configuration
├── utils/
│   ├── __init__.py
│   ├── visualization.py           # Visualization utilities
│   ├── spectral_bands.py         # Spectral band processing
│   ├── process.py                # Image processing utilities
│   └── metrics.py                # Evaluation metrics
├── data/
│   ├── __init__.py
│   ├── dataset.py                # Base dataset class
│   ├── transforms.py             # Data augmentation
│   └── datasets/                 # Dataset implementations
├── configs/                      # YAML configuration files
│   ├── train.yaml               # Training configuration
│   └── process.yaml             # Processing configuration
├── scripts/
│   ├── train.py                 # Training script
│   └── process.py               # Processing script
└── artifacts/                    # MLflow artifacts
    ├── models/                   # Saved models
    ├── metrics/                  # Logged metrics
    └── plots/                    # Generated plots
```

## Contributing

Feel free to open issues or submit pull requests for:
- Bug fixes
- New features
- Documentation improvements
- Performance optimizations

## License

This project is open-source and available under the MIT License.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{deeplabv3plus-multispectral,
  author = {OpenHands},
  title = {Multi-Spectral DeepLabV3+ Implementation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/multispectral-deeplabv3plus}
}
```

## Acknowledgments

- Original DeepLabV3+ paper: "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
- PyTorch team for the framework and pretrained models