"""Swin-Unet model implementation.

Based on the 2021 paper by Hu Cao et al.:
https://arxiv.org/pdf/2105.05537

Informed by sources:
* HuCaoFighting @ GitHub: https://github.com/HuCaoFighting/Swin-Unet
"""

__all__ = ["Swin_Unet"]

from logging    import Logger

from utils      import LOGGER

class Swin_Unet():
    """Swin-Unet (Transformer-based) segmentation model."""
    
    def __init__(self):
        """Initialize Swin-Unet model."""
        
        # Initialize logger
        __logger__: Logger =    LOGGER.getChild("swin-unet")
        
        # This model has not been implemented
        raise NotImplementedError(f"Swin-Unet model is not yet supported.")