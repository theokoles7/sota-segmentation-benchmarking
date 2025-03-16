"""U-Net model implementation.

Based on the 2015 paper by Olaf Ronneberger et al.:
https://arxiv.org/pdf/1505.04597

Informed by sources:
* Geeks4Geeks: https://www.geeksforgeeks.org/u-net-architecture-explained/#
"""

__all__ = ["UNet"]

from logging    import Logger

from utilities  import LOGGER

class UNet():
    """U-Net (CNN-based) segmentation model."""
    
    def __init__(self):
        """Initialize U-Net model."""
        
        # Initialize logger
        __logger__: Logger =    LOGGER.getChild("u-net")
        
        # This model has not been implemented
        raise NotImplementedError(f"U-Net model is not yet supported.")