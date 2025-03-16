"""Seg-Former model implementation.

Based on the 2021 paper by Enze Xie et al.:
https://proceedings.neurips.cc/paper/2021/file/64f1f27bf1b4ec22924fd0acb550c235-Paper.pdf

Informed by sources:
* NVLabs @ GitHub: https://github.com/NVlabs/SegFormer
"""

__all__ = ["Seg_Former"]

from logging    import Logger

from utilities  import LOGGER

class Seg_Former():
    """Seg-Former (Transformer-based) segmentation model."""
    
    def __init__(self):
        """Initialize Seg-Former model."""
        
        # Initialize logger
        __logger__: Logger =    LOGGER.getChild("seg-former")
        
        # This model has not been implemented
        raise NotImplementedError(f"Seg-Former model is not yet supported.")