import torch

def calculate_indices(images: torch.Tensor, indices_list: list) -> torch.Tensor:
    """Calculate spectral indices from multi-band imagery.
    
    Args:
        images: Tensor of shape [B, C, H, W] where C is number of bands
        indices_list: List of index names to calculate
        
    Returns:
        Tensor of shape [B, N, H, W] where N is number of indices
    """
    batch_size, _, height, width = images.shape
    indices_tensor = torch.zeros((batch_size, len(indices_list), height, width),
                               device=images.device)
    
    eps = 1e-8  # Small constant to avoid division by zero
    
    for i, index_name in enumerate(indices_list):
        if index_name == 'ndvi':
            # NDVI = (NIR - Red) / (NIR + Red)
            nir = images[:, 7:8]  # NIR band
            red = images[:, 3:4]  # Red band
            indices_tensor[:, i:i+1] = (nir - red) / (nir + red + eps)
            
        elif index_name == 'evi':
            # EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
            nir = images[:, 7:8]
            red = images[:, 3:4]
            blue = images[:, 1:2]
            indices_tensor[:, i:i+1] = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + eps)
            
        elif index_name == 'ndwi':
            # NDWI = (Green - NIR) / (Green + NIR)
            green = images[:, 2:3]
            nir = images[:, 7:8]
            indices_tensor[:, i:i+1] = (green - nir) / (green + nir + eps)
            
        elif index_name == 'ndbi':
            # NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
            swir1 = images[:, 8:9]
            nir = images[:, 7:8]
            indices_tensor[:, i:i+1] = (swir1 - nir) / (swir1 + nir + eps)
            
        elif index_name == 'savi':
            # SAVI = ((NIR - Red) / (NIR + Red + 0.5)) * (1.5)
            nir = images[:, 7:8]
            red = images[:, 3:4]
            indices_tensor[:, i:i+1] = ((nir - red) / (nir + red + 0.5 + eps)) * 1.5
            
        elif index_name == 'ndre':
            # NDRE = (NIR - RedEdge) / (NIR + RedEdge)
            nir = images[:, 7:8]
            red_edge = images[:, 4:5]  # Using first red edge band
            indices_tensor[:, i:i+1] = (nir - red_edge) / (nir + red_edge + eps)
            
        elif index_name == 'moisture':
            # Moisture Index = (NIR - SWIR2) / (NIR + SWIR2)
            nir = images[:, 7:8]
            swir2 = images[:, 9:10]
            indices_tensor[:, i:i+1] = (nir - swir2) / (nir + swir2 + eps)
    
    return indices_tensor