"""Multi-spectral vision processing with DeepLabV3+."""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

__author__ = "OpenHands"
__email__ = "openhands@all-hands.dev"

from .models.dual_source_deeplabv3 import DualSourceDeepLabV3Plus
from .config.mlflow_config import MLflowConfig, MLflowTracker

__all__ = [
    "DualSourceDeepLabV3Plus",
    "MLflowConfig",
    "MLflowTracker",
]