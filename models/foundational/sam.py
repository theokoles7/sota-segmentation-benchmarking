"""SAM model implementation.

Based on the 2023 paper by Alexander Kirillov et al.:
https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf

Informed by sources:
* facebookresearch @ GitHub: https://github.com/facebookresearch/segment-anything
"""

__all__ = ["SAM"]

from logging    import Logger

from utils      import LOGGER

class SAM():
    """SAM (Foundational) segmentation model."""
    
    def __init__(self):
        """Initialize SAM model."""
        
        # Initialize logger
        __logger__: Logger =    LOGGER.getChild("sam")
        
        # This model has not been implemented
        raise NotImplementedError(f"SAM model is not yet supported.")