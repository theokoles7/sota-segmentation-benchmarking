"""DeepLabV3+ model implementation.

Based on the 2018 paper by Liang-Chieh Chen et al.:
https://openaccess.thecvf.com/content_ECCV_2018/papers/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.pdf

Informed by sources:
* lattice-ai @ GitHub: https://github.com/lattice-ai/DeepLabV3-Plus
"""

__all__ = ["DeepLabV3"]

from logging    import Logger

from utils      import LOGGER

class DeepLabV3():
    """DeepLabV3+ (CNN-based) segmentation model."""
    
    def __init__(self):
        """Initialize DeepLabV3+ model."""
        
        # Initialize logger
        __logger__: Logger =    LOGGER.getChild("deeplab-v3")
        
        # This model has not been implemented
        raise NotImplementedError(f"DeepLabV3+ model is not yet supported.")