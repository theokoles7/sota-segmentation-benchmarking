"""MedSAM model implementation.

Based on the 2023 paper by Jun Ma et al.:
https://www.nature.com/articles/s41467-024-44824-z.pdf

Informed by sources:
* bowang-lab @ GitHub: https://github.com/bowang-lab/MedSAM
"""

__all__ = ["MedSAM"]

from logging    import Logger

from utilities  import LOGGER

class MedSAM():
    """MedSAM (Foundational) segmentation model."""
    
    def __init__(self):
        """Initialize MedSAM model."""
        
        # Initialize logger
        __logger__: Logger =    LOGGER.getChild("med-sam")
        
        # This model has not been implemented
        raise NotImplementedError(f"MedSAM model is not yet supported.")