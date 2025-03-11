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

```python
import torch
from models.dual_source_deeplabv3 import DualSourceDeepLabV3Plus
from config.mlflow_config import MLflowConfig, MLflowTracker
from spectral_bands import SpectralBands, SpectralIndices

# Initialize MLflow tracking
mlflow_config = MLflowConfig(
    experiment_name="multispectral_segmentation",
    tracking_uri="sqlite:///mlflow.db"
)
tracker = MLflowTracker(mlflow_config)

# Initialize model
model = DualSourceDeepLabV3Plus(
    num_classes=21,          # Number of segmentation classes
    rgb_encoder="resnet50",  # High-quality RGB encoder
    ms_encoder="resnet50",   # Multi-spectral encoder
    ms_in_channels=8,        # Number of MS bands
    fusion_channels=256      # Feature fusion channels
)

# Start MLflow run
tracker.start_run(run_name="dual_source_test")

# Log model parameters
tracker.log_params({
    "num_classes": 21,
    "rgb_encoder": "resnet50",
    "ms_encoder": "resnet50",
    "ms_in_channels": 8,
    "fusion_channels": 256
})

# Prepare your data
# Option 1: Dual-source input (RGB + MS)
rgb_data = torch.randn(2, 3, 512, 512)        # High-quality RGB
ms_data = torch.randn(2, 8, 512, 512)         # Multi-spectral bands

# Option 2: Single-source input (MS only)
ms_only_data = torch.randn(2, 8, 512, 512)    # Multi-spectral bands

# Forward pass examples
# Dual-source mode
output_dual = model(rgb_img=rgb_data, ms_img=ms_data)

# Single-source mode (MS only)
output_ms = model(ms_img=ms_only_data)

# Calculate spectral indices if needed
indices = SpectralIndices()
ndvi = indices.ndvi(ms_data)
evi = indices.evi(ms_data)

# Log metrics
tracker.log_metrics({
    "ndvi_mean": ndvi.mean().item(),
    "evi_mean": evi.mean().item()
})

# End MLflow run
tracker.end_run()
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
├── models/
│   └── dual_source_deeplabv3.py    # Dual-source DeepLabV3+ implementation
├── config/
│   └── mlflow_config.py            # MLflow configuration and utilities
├── utils/
│   ├── visualization.py            # Visualization utilities
│   └── spectral_bands.py          # Spectral band processing
├── data/
│   └── datasets/                   # Dataset implementations
├── experiments/
│   └── configs/                    # Experiment configurations
└── artifacts/                      # MLflow artifacts
    ├── models/                     # Saved models
    ├── metrics/                    # Logged metrics
    └── plots/                      # Generated plots
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